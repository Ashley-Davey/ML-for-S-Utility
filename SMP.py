
import tensorflow as tf
import numpy as np
from common import log, plot, FeedForwardSubNet, Method, Problem, Solution, penalise_range, evaluate
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

class SMPSolver(object):
    def __init__(self, config, fn = None, name = ''):
        log('Running SMP Algorithm')
        self.start_time = time.time()
        self.name = name
        self.config = config
        self.method = Method(config, 'smp')
        self.model = SMPModel(config, fn)

        
        assert np.any(np.isclose(np.linspace(0, self.config.T, self.method.bsde_N + 1), self.method.test_time))
        
        self.p_optimiser  = tf.keras.optimizers.legacy.Adam(learning_rate=self.method.learning_rate_value)
        self.control_optimiser = tf.keras.optimizers.legacy.Adam(learning_rate=self.method.learning_rate_control) 
        
    def get_sample(self, num_sample, final = False):
        
        if final:
            paths = np.random.normal(size=[num_sample * 2, 1, self.method.bsde_N]) 

            X01 = np.linspace(
                self.config.x_range_final[0],
                self.config.x_range_final[1],
                num_sample
            )[:, np.newaxis]
            X02 = np.ones(num_sample)[:, np.newaxis]

            if self.config.r_range_final[0] == self.config.r_range_final[1]:
                r = self.config.r_range_final[0]
            else:
                r = 1.0       
                
            R01 = np.ones(num_sample)[:, np.newaxis] * r
            R02 = np.linspace(
                self.config.r_range_final[0],
                self.config.r_range_final[1],
                num_sample
            )[:, np.newaxis]
            
            
            X0 = np.concatenate((X01,X02))
            R0 = np.concatenate((R01,R02))
            
            # X0 = np.random.uniform(
            #     self.config.x_range_training[0] ,
            #     self.config.x_range_training[1] ,
            #     size = [num_sample * 2, 1]
            # )
            # R0 = np.random.uniform(
            #     self.config.r_range_training[0],
            #     self.config.r_range_training[1],
            #     num_sample * 2
            # )[:, np.newaxis]
            
        else:
            paths = np.random.normal(size=[num_sample, 1, self.method.bsde_N]) 

            X0 = np.random.uniform(
                self.config.x_range_final[0] ,
                self.config.x_range_final[1] ,
                size = [num_sample, 1]
            )
            R0 = np.random.uniform(
                self.config.r_range_final[0],
                self.config.r_range_final[1],
                num_sample
            )[:, np.newaxis]
            
        dw_sample = paths * self.method.sqrt_delta_t


        return dw_sample, X0, R0
        

        
        
    def train(self):
        
        # begin sgd iteration
        
        data = {
            'control losses': [],
            'M losses': [],
            'times': [],
            }
        
        try:
            try:
                error_v = np.NaN
                error_c = np.NaN
                step=0
        
                sample_data_final = self.sample(final = True)
                    
                log(f"Step: {0:6d} \t Time: {time.time() - self.start_time:5.2f} ")
                # sample_data = self.sample()
                for step in range(1, self.method.iteration_steps + 1):
                    sample_data = self.get_sample(self.method.batch_size)

                    display = step % self.method.display_step == 0 or step == 1
                    processes, loss = self.train_step(sample_data)
                                                
                    data['control losses'].append(abs(loss['pi'].numpy()))
                    data['M losses'].append(loss['p'].numpy())
                    error_v = loss['p'].numpy()
                    error_c = abs(loss['pi'].numpy())
                    data['times' ].append(time.time() - self.start_time)
                    if display:
                        log(f"Step: {step:6d} \t Time: {time.time() - self.start_time:5.2f} \t V-C Losses: {error_v:.3e} - {error_c:.3e} ")
                        if plot_mid or (plot_early and step == 1):
                            plot(self.config, 'smp', data, processes, step)
            except Exception as e:
                    log('Termination due to ', e)
                    pass
        except BaseException:
            log('Terminated Manually')
            pass
        
        log(f"Ended at {step} iterations")
        log(f"Step: {step:6d} \t Time: {time.time() - self.start_time:5.2f} \t V-C Losses: {error_v:.3e} - {error_c:.3e}")
        
        processes, loss = self.train_step(sample_data_final)

        if plot_late:
            plot(self.config, 'smp', data, processes, self.name)

        return data, processes


    @tf.function
    def train_step(self, sample_data):
        process_data, grads, loss = self.model(sample_data)
        self.p_optimiser.apply_gradients(zip(grads['p'], self.model.subnet_p.trainable_variables))
        self.control_optimiser.apply_gradients(zip(grads['pi'], self.model.subnet_pi.trainable_variables))
        return process_data, loss
    
        
    def sample(self, final = False):
        return self.get_sample(self.method.batch_size, final)

     
class SMPModel(tf.keras.Model):
    def __init__(self, config, fn = None):
        super(SMPModel, self).__init__()
        self.config = config
        self.method = Method(config)

        self.subnet_p  = FeedForwardSubNet(dim = 1, fn = fn)
        self.subnet_pi = FeedForwardSubNet(dim = 1, fn = fn)
            
            
    def model_data(self, sample_data, training, tape):
        dw, X0, R0 = sample_data
        all_one_vec = tf.ones(shape=tf.stack([tf.shape(dw)[0], 1]), dtype=tf.float64)
        time_array = tf.range(self.method.bsde_N + 1, dtype = tf.float64) * self.method.delta_t
        time_stamp_rev = tf.ones(shape=tf.stack([tf.shape(dw)[0], self.method.bsde_N + 1]), dtype=tf.float64) * time_array 
        time_stamp = tf.transpose(time_stamp_rev)[:,:,np.newaxis]
        
        x = all_one_vec * X0 
        tape.watch(x)
        R_0 = all_one_vec * R0
        min_x = x
        W = all_one_vec * 0.0
                
        mult = self.config.x_range_final[1]

        state = tf.concat([mult * time_stamp[0], x, R_0], 1)
        
        p_0   = self.subnet_p(state) #+ self.config.dU_1(x)
        dp_0 = tape.gradient(p_0, x)
        pi = self.subnet_pi(state)
    
                
        process_data = {
            'x': x,
            'r': R_0,
            't': tf.reshape(time_stamp, tf.stack([tf.shape(dw)[0] * (self.method.bsde_N + 1), 1])),
            'pi': pi,
            'dM': -p_0,
            'dM2': -dp_0,
            }
        
        for i in range(self.method.bsde_N):
            
            x = (
                x
                + x  * self.config.alpha * self.method.delta_t 
                + pi * self.config.sigma * self.config.theta * self.method.delta_t 
                + pi * self.config.sigma * dw[:, :, i]
                )
            
            min_x = tf.minimum(x, min_x)
                        
            W = W + dw[:, :, i]
            b = np.sign(self.config.rho) * self.config.b

            R = R_0 * tf.exp((self.config.a - 0.5 * b ** 2) * time_array[i + 1] + b * W)
            p = p_0 * tf.exp(-(self.config.alpha + 0.5 * self.config.theta ** 2) * time_array[i + 1] - self.config.theta * W)
            dp = dp_0 
            
                                                                
            if i < self.method.bsde_N - 1:
                _x = x
                state = tf.concat([mult * time_stamp[i + 1], _x, R], 1)
                pi = self.subnet_pi(state)
                
            process_data['x']   = tf.concat([process_data['x'  ], x  ], 0)
            process_data['r']   = tf.concat([process_data['r'  ], R  ], 0)
            process_data['pi']  = tf.concat([process_data['pi' ], pi ], 0)
            process_data['dM']  = tf.concat([process_data['dM' ], -p], 0)
            process_data['dM2']  = tf.concat([process_data['dM2' ], -dp], 0)
                    
        
        

        
        if self.config.non_concave:
            loss_p = ( 
                tf.reduce_mean(
                    tf.square(p + self.config.u_nonconcavex(x, R))
                    + penalise_range(min_x, lower = 1e-6)
                    )
                )
        else:
            loss_p = ( 
                tf.reduce_mean(
                    tf.square(p + self.config.ux(x, R))
                    + penalise_range(min_x, lower = 1e-6)
                    )
                )    
            loss_pi = loss_p
            loss_pi += ( 
                tf.reduce_mean(
                    - self.config.u(x, R)
                    + penalise_range(min_x, lower = 1e-6)
                    )
                )       
        
        return process_data, {
            'p': loss_p,
            'pi': loss_pi,
            }

    def call(self, sample_data, training):
        

        with tf.GradientTape(persistent=True) as tape:
            process_data, loss = self.model_data(sample_data, training, tape = tape)

        grads = {
            'p': tape.gradient(loss['p'], self.subnet_p.trainable_variables),
            'pi': tape.gradient(loss['pi'], self.subnet_pi.trainable_variables),
            }
                
        del tape
        return process_data, grads, loss
  

    
    
def main():
        
    config = Problem()
    tf.keras.backend.clear_session()
    tf.keras.backend.set_floatx('float64')
    SMPSolver(config).train()

        
                        


if __name__ == '__main__':
    plot_early = False
    plot_mid = False
    plot_late = True
    main()
    tf.keras.backend.clear_session()

    
    
    

         
