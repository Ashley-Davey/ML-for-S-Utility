
import tensorflow as tf
import numpy as np
from common import log, plot, FeedForwardSubNet, Method, Problem, Solution, penalise_range
import time
from matplotlib import pyplot as plt, rcParams


rng = np.random.default_rng()
tf .get_logger().setLevel('ERROR') #remove warning messages for taking second derivatives
rcParams['figure.dpi'] = 600

#method variables

plot_early = False
plot_mid = False
plot_late = False







    
            
#solver

class PINNSolver(object):
    def __init__(self, config, fn = None, name = ''):
        log(f"Running {'non-concave ' if config.non_concave else ''}PINN Algorithm")
        self.config = config
        self.start_time = time.time()
        self.model = PINNModel(config, fn)
        self.method = Method(config, 'pinn')
        self.activation_function = name
        
        self.M_optimiser = tf.keras.optimizers.Adam(learning_rate=self.method.learning_rate_value)
        self.boundary_data = self.get_sample(self.method.num_sample_bound, 'boundary')
        self.time_data = self.get_sample(self.method.num_sample_bound, 'time')
        self.collocation_data = self.get_sample(self.method.num_sample_coll, 'collocation')
        self.boundary_data_final = self.get_sample(self.method.num_sample_bound, 'boundary', final = True)
        self.collocation_data_final = self.get_sample(self.method.num_sample_coll, 'collocation',  final = True)
        self.time_data_final = self.get_sample(self.method.num_sample_bound, 'time', final = True)

    def get_sample(self, size, area, final = False):
        
        if final:
            x_points = np.linspace(self.config.x_range_final[0], self.config.x_range_final[1], size)
        else:
            x_points = rng.uniform(self.config.x_range_training[0], self.config.x_range_training[1], size)
            
        if area == 'boundary':
            t_points = np.ones(size) * self.config.T
        elif area == 'collocation':
            if final:
                t_points = np.ones(size) * self.method.test_time
                # t_points = rng.uniform(0, self.config.T, size) 

            else:
                t_points = rng.uniform(0, self.config.T, size) 
        elif area == 'time':
                t_points = np.linspace(0, self.config.T, size) 
                x_points = np.ones(size) * self.config.x_range_training[0]
        else:
            raise ValueError('Bad area')
            
        return np.stack([t_points, x_points], -1)

        
            
    def train(self):
        # begin sgd iteration
                
        data = {
            'M losses': [],
            'times': [],
            }
        
        try:
            try:
                error_v = np.NaN
                min_loss = np.inf
                step=0  
                
                
                log(f"Step: {0:6d} \t Time: {time.time() - self.start_time:5.2f} ")
                sample_data = self.sample()
                sample_data_final = self.sample(final = True)
                while step < self.method.iteration_steps:
                    step+=1
                    display = step % self.method.display_step == 0
                    
                    
                    
                    processes, boundary_function_data, loss = self.train_step(sample_data)
                    
                    
                                                
                    data['M losses'].append(loss['M'].numpy())
                    if not display and step < self.method.iteration_steps:
                        error_v = loss['M'].numpy()
                        min_loss = min(min_loss, error_v)
                        
                    if error_v < 1e-5 and self.method.iteration_steps > 10000:
                        raise ValueError('Low Loss')
                                            
                    if error_v > 10 * min_loss and step > self.method.iteration_steps / 4 and self.method.iteration_steps > 10000:
                        log(f'Spike at step {step}: {min_loss:.3e} -> {error_v:.3e}')
                        step = int(step - self.method.iteration_steps / 5)
                        min_loss = error_v
                        
        
        
        
                    data['times' ].append(time.time() - self.start_time)
                    
        
                    if display or step == 1:
                        log(f"Step: {step:6d} \t Time: {time.time() - self.start_time:5.2f} \t Loss: {error_v:.3e}")
                        
                        if plot_mid or (plot_early and step == 1):
                            plot(self.config, 'PINN', data, processes, step)

            except Exception as e:
                log('Termination due to', e)
                pass
        except BaseException:
            log('Terminated Manually')
            pass
        
        log(f"Ended at {step} iterations")
        log(f"Step: {step:6d} \t Time: {time.time() - self.start_time:5.2f} \t Loss: {error_v:.3e}")

        processes, boundary_function_data, loss = self.train_step(sample_data_final)
        error_v = loss['M'].numpy()

        log(f"Test Data Loss: {error_v:.3e}")
        if plot_late:
            plot(self.config, 'pinn', data, processes, self.activation_function)

        return data, processes, boundary_function_data


    @tf.function
    def train_step(self, sample_data):
        collocation_function_data, boundary_function_data, grads, loss = self.model(sample_data)

        self.M_optimiser.apply_gradients(zip(grads['M'], self.model.subnet_M.trainable_variables))
        return collocation_function_data, boundary_function_data, loss
    
    def sample(self, final = False):
        if final:
            return {
                'collocation': self.collocation_data_final,
                'boundary': self.boundary_data_final,
                'time': self.time_data_final
                } 
        else:
            return {
                'collocation': self.collocation_data,
                'boundary': self.boundary_data,
                'time': self.time_data
                } 

class PINNModel(tf.keras.Model):
    def __init__(self, config, optimise_control = False, iterative = False, fn = None):
        super(PINNModel, self).__init__()
        self.subnet_M  = FeedForwardSubNet(fn = fn)
        self.config = config
        self.R = tf.Variable(1.0, trainable = False, dtype = tf.float64)
                    
    def collocation_loss(self, data, training, tape):
        
        t_data = data[:, :1]
        x_data = data[:,1:2]
                          
        tape.watch(t_data)
        tape.watch(x_data)
        tracked_data = tf.concat([t_data , x_data], 1)  
        M = self.subnet_M(tracked_data, training)
        
        Mt = tape.gradient(M, t_data)
        Mx = tape.gradient(M, x_data)
        
        Mxx = tape.gradient(Mx, x_data)
        
        A = self.config.control_denominator(Mx, Mxx, x_data)
        B = self.config.control_numerator  (Mx, Mxx, x_data)
            
        pi = - B / A
        pi = tf.clip_by_value(pi, self.config.pi_min, self.config.pi_max)
                
        LM = self.config.L(Mt, Mx, Mxx, x_data, pi)
        
        loss_M = tf.reduce_mean(
            tf.square(LM)
            # + penalise_range(Mxx, -np.inf, -1e-4)
            # + penalise_range(Mt, -np.inf, 0)
            ) 
        
        # loss_M = 0.1 * tf.reduce_mean(
        #     tf.square(M - self.config.u_nonconcave(x_data, self.R ))
        #     + tf.square(Mx  - self.config.u_nonconcavex(x_data, self.R))
        #     ) 
                        
        loss = {
            'M' : loss_M,
            }
        with tape.stop_recording():
            function_data = {
                't'  : t_data,
                'x'  : x_data,
                'M'  : M,
                'Mt' : Mt,
                'dM' : Mx,
                'dM2': Mxx,
                'pi' : pi      
                }
            
        return loss, function_data
    
    def time_loss(self, data, training, tape):
        
        loss = {}
        t_data = data[:, :1]
        x_data = data[:,1:2]
                          
        tape.watch(t_data)
        tape.watch(x_data)
        tracked_data = tf.concat([t_data , x_data], 1)  
        M = self.subnet_M(tracked_data, training)
        
                    
        loss_time =  tf.reduce_mean(
            tf.square(M - self.config.u(x_data, self.R ))
            ) 
        loss['M'] = loss_time                
                    
        return loss
    
    def boundary_loss(self, data, training, tape):
        
        loss = {}
        t_data = data[:, :1]
        x_data = data[:,1: ]
        
        tape.watch(t_data)
        tape.watch(x_data)
        tracked_data = tf.concat([t_data, x_data], 1)            
        M = self.subnet_M(tracked_data, training)
        Mx  = tape.gradient(M , x_data)
        Mxx = tape.gradient(Mx, x_data)
        

        if self.config.non_concave:
            loss_M = 0.1 * tf.reduce_mean(
                tf.square(M - self.config.u_nonconcave(x_data, self.R ))
                + tf.square(Mx  - self.config.u_nonconcavex(x_data, self.R))
                ) 
        else:
            loss_M = 0.1 * tf.reduce_mean(
                tf.square(M - self.config.u(x_data, self.R ))
                + tf.square(Mx  - self.config.ux (x_data, self.R ))
                ) 
        
        loss['M'] = loss_M                
        
        with tape.stop_recording():
            function_data = {
                't'  : t_data,
                'x'  : x_data,
                'M'  : M,
                'dM' : Mx,
                'dM2': Mxx,
                }
            
        return loss, function_data

    def call(self, sample_data, training):
        with tf.GradientTape(persistent=True) as tape:
            collocation_data   = tf.convert_to_tensor(sample_data['collocation'], dtype=tf.float64)
            collocation_loss, collocation_function_data = self.collocation_loss(collocation_data, training, tape)
            loss = {
                'M' : collocation_loss['M'],
                }
            
            if 'boundary' in sample_data.keys():
                boundary_data = tf.convert_to_tensor(sample_data['boundary'], dtype=tf.float64)
                boundary_loss, boundary_function_data = self.boundary_loss(boundary_data,training, tape)
                loss['M'] += boundary_loss['M']
                
            if 'time' in sample_data.keys() and self.config.x_range_training[0] < 0.1:
                time_data = tf.convert_to_tensor(sample_data['time'], dtype=tf.float64)
                time_loss = self.time_loss(time_data,training, tape)   
                loss['M']  += time_loss['M'] 
                    
            grads = {
                'M' : tape.gradient(loss['M' ], self.subnet_M .trainable_variables),
                }     
            


            
            return collocation_function_data, boundary_function_data, grads, loss

    

    
        
def main():

    for rho in [1.0, 0.0 ]:
    
        config = Problem(rho = rho, non_concave = False)
        config_nonconcave = Problem(rho = rho, non_concave = True)

        tf.keras.backend.clear_session()
        tf.keras.backend.set_floatx('float64')
        PINNSolver(config).train()
        # PINNSolver(config_nonconcave).train()
                
                        

        

    
if __name__ == '__main__':
    plot_mid = False
    plot_late = True
    tf.keras.backend.clear_session()
    main()
    tf.keras.backend.clear_session()

    
    
    

         
