# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 16:01:56 2024

@author: Charlotte
"""
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

class PINNDual_generalSolver(object):
    def __init__(self, config, fn = None, name = ''):
        log('Running PINN Dual Algorithm (without scaling)')
        self.start_time = time.time()
        self.config = config
        self.model = PINNDual_generalModel(config, fn)
        self.method = Method(config, 'dual_general')
        self.activation_function = name
        
        self.M_optimiser = tf.keras.optimizers.Adam(learning_rate=self.method.learning_rate_value)
        self.boundary_data = self.get_sample(self.method.num_sample_bound, 'boundary')
        self.collocation_data = self.get_sample(self.method.num_sample_coll, 'collocation')
        self.boundary_data_final = self.get_sample(self.method.num_sample_bound, 'boundary', final = True)
        self.collocation_data_final = self.get_sample(self.method.num_sample_coll, 'collocation',  final = True)

    def get_sample(self, size, area, final = False):
        
        if final:
            y_points = np.linspace(self.config.y_range[0], self.config.y_range[1], size)
            r_points = np.ones(size) 
        else:
            y_points = rng.uniform(self.config.y_range[0], self.config.y_range[1], size)
            r_points = rng.uniform(self.config.r_range[0], self.config.r_range[1], size)
            
        if area == 'boundary':
            t_points = np.ones(size) * self.config.T
        elif area == 'collocation':
            if final:
                t_points = np.ones(size) * self.method.test_time
                # t_points = rng.uniform(0, self.config.T, size) 
            else:
                t_points = rng.uniform(0, self.config.T, size) 
        else:
            raise ValueError('Bad area')
            
        return np.stack([t_points, y_points, r_points], -1)

        
            
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
                    display = step % self.method.display_step == 0 or step == 1
                    processes, boundary_function_data, loss = self.train_step(sample_data_final if display else sample_data)
                                                
                    data['M losses'].append(loss['M'].numpy())
                    error_v = loss['M'].numpy()
                    min_loss = min(min_loss, error_v)
                    data['times' ].append(time.time() - self.start_time)
                    
                    if error_v < 5e-5:
                        raise ValueError('Low Loss')
                        
                    if error_v > 100 * min_loss and step > self.method.iteration_steps / 4:
                        log(f'Spike at step {step}: {min_loss:.3e} -> {error_v:.3e}')
                        step = int(step - self.method.iteration_steps / 5)
                        min_loss = error_v
                        
        
                    if step % self.method.display_step == 0 or step == 1:
                        log(f"Step: {step:6d} \t Time: {time.time() - self.start_time:5.2f} \t Loss: {error_v:.3e}")

                        if plot_mid or (plot_early and step == 1):
                            plot(self.config, 'pinn dual', data, processes, step)


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
            plot(self.config, 'pinn dual', data, processes, self.activation_function)

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
                } 
        else:
            return {
                'collocation': self.collocation_data,
                'boundary': self.boundary_data,
                } 

class PINNDual_generalModel(tf.keras.Model):
    def __init__(self, config, fn = None):
        super(PINNDual_generalModel, self).__init__()
        self.config = config
        self.subnet_M  = FeedForwardSubNet(fn = fn)
        self.lam = tf.Variable(self.config.lam, trainable = False, dtype = tf.float64)
        self.L = tf.Variable(self.config.lower, trainable = False, dtype = tf.float64)

    def collocation_loss(self, data, training, tape):
        
        t_data = data[:, :1]
        y_data = data[:,1:2]
        r_data = data[:,2:3]
                          
        tape.watch(t_data)
        tape.watch(y_data)
        tape.watch(r_data)
        tracked_data = tf.concat([t_data, y_data, r_data], 1)  
        M = self.subnet_M(tracked_data, training)
        
        Mt = tape.gradient(M, t_data)
        My = tape.gradient(M, y_data)
        Mr = tape.gradient(M, r_data)
        
        Myy = tape.gradient(My, y_data)
        Myr = tape.gradient(My, r_data)
        Mrr = tape.gradient(Mr, r_data)
                                    
                        
        LM = self.config.L_dual_r(Mt, My, Mr, Myy, Myr, Mrr, y_data, r_data)
        
        loss_M = tf.reduce_mean(
            tf.square(LM)
            + penalise_range(Myy, 1e-1, np.inf)
            + penalise_range(My, -np.inf, -1e-3)
            ) 
        
                
        loss = {
            'M' : loss_M,
            }

        
        A = self.config.control_denominator_r(y_data, Mr,  -1 / Myy, 0, Mrr, -My, r_data)
        B = self.config.control_numerator_r  (y_data, Mr,  -1 / Myy, 0, Mrr, -My, r_data)
            
        pi = - B / A
        pi = tf.clip_by_value(pi, self.config.pi_min, self.config.pi_max)
        
        with tape.stop_recording():
            function_data = {
                't'  : t_data,
                'x'  : -My,
                'M'  : M - My * y_data ,
                'dM' : y_data,
                'dM2': -1 / Myy,
                'pi' :  pi,   
                'Mt_dual' : Mt
                }
            
        return loss, function_data
    
    
    def boundary_loss(self, data, training, tape):
        
        loss = {}
        t_data = data[:, :1]
        y_data = data[:,1:2]
        r_data = data[:,2:3]
        
        tape.watch(t_data)
        tape.watch(y_data)
        tracked_data = tf.concat([t_data, y_data, r_data], 1)            
        M = self.subnet_M(tracked_data, training)
        My  = tape.gradient(M , y_data)

        loss_M = 0.1 * tf.reduce_mean(
            tf.square(M - self.config.u_tilde(y_data, self.lam, self.L, r_data))
            + tf.square(My  - self.config.u_tildey(y_data, self.lam, self.L, r_data))
            ) 
        
        loss['M'] = loss_M                
        
        with tape.stop_recording():
            function_data = {
                'y'  : y_data,
                'M_dual'  : M,
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
                
            grads = {
                'M' : tape.gradient(loss['M' ], self.subnet_M .trainable_variables),
                }     
            
            return collocation_function_data, boundary_function_data, grads, loss

    

    
        
def main():
    for rho in [ 1.0 ]:
    
        config = Problem(rho = rho)
        tf.keras.backend.clear_session()
        tf.keras.backend.set_floatx('float64')
        solver = PINNDual_generalSolver(config)
        data, processes, boundary_processes = solver.train()

        
 

    
if __name__ == '__main__':
    plot_early = False
    plot_mid = False
    plot_late = True

    main()
    tf.keras.backend.clear_session()

    


