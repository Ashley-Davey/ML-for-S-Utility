# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 17:29:14 2024

@author: ashle
"""

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt, rcParams
from scipy.optimize import minimize
from scipy.stats import norm
import time

rcParams['figure.dpi'] = 600
tf .get_logger().setLevel('ERROR') #remove warning messages for taking second derivatives

#config variables

test_time = 0.0
sample_size = 1000
solution_size = 100

monte_size = 10000

monte_M = 50
monte_N = 50

num_hiddens = [50, 50]
threshold = -15


pdf = norm.pdf
cdf = norm.cdf

def log(*args, **kwargs):
    now = time.strftime("%H:%M:%S")
    print("[" + now + "] ", end="")
    print(*args, **kwargs)
    
def quadratic(a,b,c):
    discriminant = b ** 2 - 4 * a * c
    sqrt_term = tf.sqrt(discriminant + 1e-8)
    
    return tf.where (
        -b - sqrt_term > 0,
        (-b - sqrt_term) / (2 * a),
        (-b + sqrt_term) / (2 * a)
        )

def quadratic_np(a,b,c):
    discriminant = b ** 2 - 4 * a * c
    sqrt_term = np.sqrt(np.maximum(discriminant, 1e-8))
    
    return np.where (
        -b - sqrt_term > 0,
        (-b - sqrt_term) / (2 * a),
        (-b + sqrt_term) / (2 * a)
        )

class Method(object):
    def __init__(self, config, name = 'pinn'):
        self.learning_rate_value_init = {
            'pinn': 0.0001,
            'pinn_general': 0.0001,
            'dual': 0.00005,# * (10 ** (np.isclose(np.abs(config.rho), 1.0))),
            'dual_general': 0.00005,# * (10 ** (np.isclose(np.abs(config.rho), 1.0))),
            'smp':  0.01,
            'bsde': 0.01
            }[name]
        self.learning_rate_control_init = 0.01
        self.learning_rate_lam = 0.01
        self.iteration_steps = {
            'pinn': 50000,
            'pinn_general': 50000,
            'dual': 50000,
            'dual_general': 50000,
            'smp':  2000,
            'bsde': 500
            }[name]
        self.drops = 0
        self.display_step = max(int(self.iteration_steps / 2), 1)
        self.learning_rate_value, self.learning_rate_control = self.learning_rates(1)
        self.batch_size = 500
        self.final_batch_size = self.batch_size
        self.test_time = test_time
        self.num_sample_coll = 1000
        self.num_sample_bound = 100
        self.sample_size = sample_size
        self.bsde_N = 100
        self.delta_t = config.T / self.bsde_N
        self.sqrt_delta_t = np.sqrt(self.delta_t)

    
    def learning_rates(self, m):
        if self.drops > 0:
            boundaries    = [m * self.iteration_steps * 0.5 ** n for n in range(self.drops, 0, -1) ]
            control_rates = [0.1 ** n * self.learning_rate_control_init for n in range(self.drops + 1)]
            value_rates   = [0.1 ** n * self.learning_rate_value_init   for n in range(self.drops + 1)]
        
            learning_rate_value = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                boundaries, value_rates
            )
                    
            learning_rate_control = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                boundaries, control_rates
            )
            log( 'Learning Rate Schedule: ', boundaries, value_rates)
        else:
            learning_rate_value = self.learning_rate_value_init
            learning_rate_control = self.learning_rate_control_init
        return learning_rate_value, learning_rate_control

class Problem(object):
    def __init__(self, R = 1.0, lower = 0.0, lam = 0.0, non_concave = False, rho = 1.0, plot = False, scaling = True):
    
        self.pi_max = 500.0 
        self.pi_min = 0.01
        self.train_lam = False
        self.non_concave = non_concave
        # self.x_max = 20
        
        self.alpha = 0.05
        self.sigma = 0.2
        self.mu    = 0.1
        
        self.p = 0.5
        assert self.p == 0.5
        self.scaling = scaling
        
        self.theta = (self.mu - self.alpha) / self.sigma
        
        

                
        
        self.T = 0.5
        
        self.R = R
        
        self.lower = lower
        self.lam = lam
        
        self.K = 0.5
        
        self.a = 0.03
        self.b = 0.1
        self.rho = rho
        
        self.La = 0.0
        self.Lb = 0.0
        
        self.epsilon = 0.0 # p(X > L) >=  1 - epsilon

        
        assert min(self.lower, self.lam) * min(self.a, self.b) == 0
        
        self.theta_bar = self.theta + self.rho * self.b * (self.p - 1)
        self.alpha_bar = ( 
            self.alpha - self.a - (self.p - 1) * self.b ** 2
            + self.b * self.rho * self.theta_bar
            )
        self.discount = self.p * (self.a + 0.5 * (self.p - 1) * self.b ** 2) 
        # log(self.theta_bar, self.alpha_bar)
        self.risk = self.theta_bar / (self.sigma * (1 - self.p))
        
        # log(self.risk)
        
        self.x_range = (0.1, 5.0)#max(self.R * 5.0, self.lower * 1.5, 5.0))
        self.x_sample = ( self.x_range[0], 1.5 * self.x_range[1])
        self.y_range = (0.25, 1.0)
        self.r_range = (0.8, 1.2)



        
        if plot:
            xaxis = np.linspace(self.x_range[0], self.x_range[1], 1000)
            
            
            u = self.u_nonconcave_np(xaxis, self.lam, self.lower, self.R)
            u_conv = self.u_np(xaxis, self.lam, self.lower, self.R)
            du_conv = self.ux_np(xaxis, self.lam, self.lower, self.R)
            
            plt.figure()
        
            
            plt.axvline(self.R, label = '$R$', linestyle = 'dotted', c = 'C3')
            plt.axvline(self.lower, label = '$L$', linestyle = 'dotted', c = 'C4')
            
            if self.lower >= self.x_hat_np(self.R):
                if self.k_np(self.lam, self.lower, self.R) > self.c_np(self.x_hat_np(self.R), self.R):
                    pass
                else:
                    plt.axvline(self.L_0_np(self.lam, self.lower, self.R), label = '$L_0$', linestyle = 'dotted', c = 'C6')
                    plt.axvline(self.x_hat_np(self.R), label = '$z$', linestyle = 'dotted', c = 'C5')
            elif self.lower >= self.R:
                if self.k_np(self.lam, self.lower, self.R) >= self.c_np(self.lower, self.R):
                    pass
                else:
                    plt.axvline(self.z_0_np(self.lam, self.lower, self.R), label = '$z_0$', linestyle = 'dotted', c = 'C6')
            else:
                if self.k_np(self.lam, self.lower, self.R) > self.c_np(self.z_tilde_np(self.lower, self.R), self.R):
                    plt.axvline(self.z_tilde_np(self.lower, self.R), label = '$\\tilde{z}$', linestyle = 'dotted', c = 'C6')
                else:
                    plt.axvline(self.z_tilde_0_np(self.lam, self.lower, self.R), label = '$\\tilde{z}_0$', linestyle = 'dotted', c = 'C7')
            
            
            plt.plot(xaxis, u, label = '$U$')
            plt.plot(xaxis, u_conv, linestyle = 'dashed', label = '$\\bar{U}$')
            plt.grid(0.5)
            plt.xlabel('$x$')
            plt.ylabel('$U$')
            plt.title(f'Utility Function (R={self.R:.2f}, L={self.lower:.2f}, $\lambda=${self.lam:.2f})')
            plt.legend()
            
            
            plt.figure()
        
            
            plt.axvline(self.R, label = '$R$', linestyle = 'dotted', c = 'C3')
            plt.axvline(self.lower, label = '$L$', linestyle = 'dotted', c = 'C4')
            
            if self.lower >= self.x_hat_np(self.R):
                if self.k_np(self.lam, self.lower, self.R) > self.c_np(self.x_hat_np(self.R), self.R):
                    pass
                else:
                    plt.axvline(self.L_0_np(self.lam, self.lower, self.R), label = '$L_0$', linestyle = 'dotted', c = 'C6')
                    plt.axvline(self.x_hat_np(self.R), label = '$z$', linestyle = 'dotted', c = 'C5')
            elif self.lower >= self.R:
                if self.k_np(self.lam, self.lower, self.R) >= self.c_np(self.lower, self.R):
                    pass
                else:
                    plt.axvline(self.z_0_np(self.lam, self.lower, self.R), label = '$z_0$', linestyle = 'dotted', c = 'C6')
            else:
                if self.k_np(self.lam, self.lower, self.R) > self.c_np(self.z_tilde_np(self.lower, self.R), self.R):
                    plt.axvline(self.z_tilde_np(self.lower, self.R), label = '$\\tilde{z}$', linestyle = 'dotted', c = 'C6')
                else:
                    plt.axvline(self.z_tilde_0_np(self.lam, self.lower, self.R), label = '$\\tilde{z}_0$', linestyle = 'dotted', c = 'C7')
            
            
            plt.plot(xaxis, du_conv, linestyle = 'dashed', label = '$\partial_x\\bar{U}$', c = 'C1')
            plt.grid(0.5)
            plt.xlabel('$x$')
            plt.ylabel('$\partial_x U$')
            plt.title(f'Derivative of Utility Function (R={self.R:.2f}, L={self.lower:.2f}, $\lambda=${self.lam:.2f})')
            plt.legend()
            plt.show()
        
        if self.lower + self.lam > 0:
            if self.lower >= self.x_hat_np(self.R):
                if self.k_np(self.lam, self.lower, self.R) > self.c_np(self.x_hat_np(self.R), self.R):
                    log('case1a')
                else:
                    log('case1b')
            elif self.lower >= self.R:
                if self.k_np(self.lam, self.lower, self.R) >= self.c_np(self.lower, self.R):
                    log('case2a')
                else:
                    log('case2b')            
            else:
                if self.k_np(self.lam, self.lower, self.R) > self.c_np(self.z_tilde_np(self.lower, self.R), self.R):
                    log('case3a')
                else:
                    log('case3b')

    
        
                        
        
    ### TENSORFLOW
                
            
    def U_1(self, z):
        return tf.pow(tf.maximum(z, 1e-8), self.p)
    
    def dU_1(self, z):
        return self.p * tf.pow(tf.maximum(z, 1e-8), self.p - 1)
    
    def d2U_1(self, z):
        return self.p * (self.p - 1) * tf.pow(tf.maximum(z, 1e-8), self.p - 2)
    
    def c(self, z, R):
        return self.dU_1(z - R)
    
    def U_2(self, z):
        if self.scaling:
            return self.K * tf.pow(tf.maximum(z, 1e-8), self.p)
        else:
            return self.K * tf.math.log(tf.maximum(z, 1e-8) + 1)
    
    def dU_2(self, z):
        if self.scaling:
            return self.p * self.K * tf.pow(tf.maximum(z, 1e-8), self.p - 1)
        else:
            return self.K / (tf.maximum(z, 1e-8) + 1)
    
    def utility(self, x, R):
        return tf.where(x > R,
                            self.U_1(x - R),
                            - self.U_2(R - x)
                            )

    def utilityx(self, x, R):
        return tf.where(x > R,
                            self.dU_1(x - R),
                            self.dU_2(R - x)
                            )
    
    

    def k(self, lam, L, R):
        return (self.utility(L, R) + lam  + self.U_2(R)) / tf.maximum(L, 1e-1)
    
    def x_hat(self, R):
        return tf.where(R > 0,
            R + (tf.sqrt(self.K ** 2 * tf.pow(R, 2 * self.p) + R) - self.U_2(R)) ** 2,
            0.0 * R
            )
            
    def L_0(self, lam, L, R):
        return tf.where(
            L >= self.x_hat(R),
            tf.where(
                self.k(lam, L, R) > self.c(self.x_hat(R), R),
                R * 0.0,
                (
                    R
                    + quadratic(
                        0.5,
                        -(tf.sqrt(L - R) + lam),
                        0.5 * (L - R)
                        ) ** 2
                    )
            ),
            R * 0.0
            )
                    
    def z_0(self, lam, L, R):
        return tf.where(
            L >= R,
            tf.where(
                self.k(lam, L, R) > self.c(L, R),
                R * 0.0,
                (
                    R
                    + quadratic(
                        0.5,
                        self.U_2(R) + lam,
                        - 0.5 * R
                        ) ** 2
                    )
                ),
             lam * 0.0
             )

            
    def z_tilde(self, L, R):
        return tf.where(
            L >= R,
            R * 0.0,
            (
                R
                + quadratic(
                    0.5,
                    self.U_2(R - L),
                    0.5 * (L - R)
                    ) ** 2
                )
            )
        
    def z_tilde_0(self, lam, L, R):
        return tf.where(
            L >= R,
            R * 0.0,
            tf.where( 
               self.k(lam, L, R) > self.c(self.z_tilde(L, R), R),
               R * 0.0,
               (
                    R
                    + quadratic(
                        0.5,
                        self.U_2(R) + lam,
                        - 0.5 * R
                        ) ** 2
                    )
               )
            )

    def u_nonconcave(self, x, lam, L, R):
        return (
            self.utility(x, R)
            + lam * tf.where(x > L, tf.ones_like(x), tf.zeros_like(x))
            )    
    
    def u_nonconcavex(self, x, R):
        return (
            self.utilityx(x, R)
            ) 
                   
    def u(self, x, lam, L, R):
        return tf.where(
            L >= self.x_hat(R),
            tf.where( 
                self.k(lam, L, R) > self.c(self.x_hat(R), R),
                ( 
                    tf.where(x >= L,
                                self.U_1(x-R) + lam, 
                                self.k(lam, L, R) * x - self.U_2(R)
                                )
                    ),
                ( 
                    tf.where(x >= L,
                                self.U_1(x-R) + lam, 
                                tf.where(x >= self.L_0(lam, L, R),
                                         self.c(self.L_0(lam, L, R), R) * (x - L) + self.U_1(L - R) + lam,
                                         tf.where(x >= self.x_hat(R),
                                                  self.U_1(x - R),
                                                  self.c(self.x_hat(R), R) * x - self.U_2(R)
                                                  )
                                         )
                                )
                    )
                ),
            tf.where (
                L >= R,
                tf.where( 
                    self.k(lam, L, R) >= self.c(L, R),
                    ( 
                        tf.where(x >= L,
                                    self.U_1(x-R) + lam, 
                                    self.k(lam, L, R) * x - self.U_2(R)
                                    )    
                        ),
                    ( 
                        tf.where(x >= self.z_0(lam, L, R),
                                    self.U_1(x - R) + lam,
                                    self.c(self.z_0(lam, L, R), R) * x - self.U_2(R)
                                    )
                        )
                ),
                tf.where(
                    self.k(lam, L, R) > self.c(self.z_tilde(L, R), R),
                    ( 
                        tf.where(x >= self.z_tilde(L, R),
                                    self.U_1(x - R) + lam,
                                    tf.where(x >= L,
                                             self.c(self.z_tilde(L, R), R) * (x - L) - self.U_2(R - L) + lam,
                                             self.k(lam, L, R) * x - self.U_2(R)
                                             )
                                    )
                        ),
                    ( 
                        tf.where(x >= self.z_tilde_0(lam, L, R),
                                    self.U_1(x - R) + lam,
                                    self.c(self.z_tilde_0(lam, L, R), R) * x - self.U_2(R)
                                    )
                        )
                    )
                )
            )
    
    def ux(self, x, lam, L, R):
        epsilon = 1e-8
        return tf.where(
            L >= self.x_hat(R),
            tf.where( 
                self.k(lam, L, R) > self.c(self.x_hat(R), R),
                ( 
                    tf.where(x >= L,
                                self.dU_1(x-R), 
                                self.k(lam, L, R) + epsilon * (L - x)
                                )
                    ),
                ( 
                    tf.where(x >= L,
                                self.dU_1(x-R), 
                                tf.where(x >= self.L_0(lam, L, R),
                                         self.c(self.L_0(lam, L, R), R),
                                         tf.where(x >= self.x_hat(R),
                                                  self.dU_1(x - R),
                                                  self.c(self.x_hat(R), R) + epsilon * (self.x_hat(R) - x)
                                                  )
                                         )
                                )
                    )
                ),
            tf.where (
                L >= R,
                tf.where( 
                    self.k(lam, L, R) >= self.c(L, R),
                    ( 
                        tf.where(x >= L,
                                    self.dU_1(x-R), 
                                    self.k(lam, L, R) + epsilon * (L - x)
                                    )    
                        ),
                    ( 
                        tf.where(x >= self.z_0(lam, L, R),
                                    self.dU_1(x - R),
                                    self.c(self.z_0(lam, L, R), R) + epsilon * (self.z_0(lam, L, R) - x)
                                    )
                        )
                ),
                tf.where(
                    self.k(lam, L, R) > self.c(self.z_tilde(L, R), R),
                    ( 
                        tf.where(x >= self.z_tilde(L, R),
                                    self.dU_1(x - R),
                                    tf.where(x >= L,
                                             self.c(self.z_tilde(L, R), R),
                                             self.k(lam, L, R) + epsilon * (L - x) 
                                             )
                                    )
                        ),
                    ( 
                        tf.where(x >= self.z_tilde_0(lam, L, R),
                                    self.dU_1(x - R),
                                    self.c(self.z_tilde_0(lam, L, R), R) + epsilon * (self.z_tilde_0(lam, L, R) - x)
                                    )
                        )
                    )
                )
            )            
    
            
    def I(self, y):
        return tf.pow(y / self.p, 1 / (self.p - 1))
            
            
    def u_tilde(self, y, lam, L, R):
        return self.u(self.x(y, lam, L, R), lam, L, R) - y * self.x(y, lam, L, R)
    
    def u_tildey(self, y, lam, L, R):
        return - self.x(y, lam, L, R)

    def x(self, y, lam, L, R):
        return tf.where(
            L >= self.x_hat(R),
            tf.where( 
                self.k(lam, L, R) > self.c(self.x_hat(R), R),
                ( 
                    tf.where(y >= self.k(lam, L, R),
                                    0.0 * y,
                                    tf.where(y >= self.c(L, R),
                                             L * tf.ones_like(y),
                                             R + self.I(y)                                  
                                             )
                                    )
                    ),
                ( 
                    tf.where(y >= self.c(self.x_hat(R), R),
                                    0.0 * y,
                                    tf.where(y >= self.c(self.L_0(lam, L, R), R),
                                             R + self.I(y),
                                             tf.where(y >= self.c(L, R),
                                                      L * tf.ones_like(y),
                                                      R + self.I(y)
                                                      )
                                             )
                                    )
                    )
                ),
            tf.where (
                L >= R,
                tf.where( 
                    self.k(lam, L, R) >= self.c(L, R),
                    ( 
                        tf.where(y >= self.k(lam, L, R),
                                        0.0 * y,
                                        tf.where(y >= self.c(L, R),
                                                 L * tf.ones_like(y),
                                                 R + self.I(y)                                  
                                                 )
                                        ) 
                        ),
                    ( 
                        tf.where(y >= self.c(self.z_0(lam, L, R), R),
                                        0.0 * y,
                                        R + self.I(y)
                                        )
                        )
                ),
                tf.where(
                    self.k(lam, L, R) > self.c(self.z_tilde(L, R), R),
                    ( 
                        tf.where(y >= self.k(lam, L, R),
                                        y * 0.0,
                                        tf.where(y >= self.c(self.z_tilde(L, R), R),
                                                 L * tf.ones_like(y),
                                                 R + self.I(y)
                                                 )
                                        )
                        ),
                    ( 
                        tf.where(y >= self.c(self.z_tilde_0(lam, L, R), R),
                                        y * 0.0,
                                        R + self.I(y)
                                        ) 
                        )
                    )
                )
            )    
    
            
    ### NUMPY
                
            
    def U_1_np(self, z):
        return np.power(np.maximum(z, 1e-8), self.p)
    
    def dU_1_np(self, z):
        return self.p * np.power(np.maximum(z, 1e-8), self.p - 1)
    
    def d2U_1_np(self, z):
        return self.p * (self.p - 1) * np.power(np.maximum(z, 1e-8), self.p - 2)
    
    def c_np(self, z, R):
        return self.dU_1_np(z - R)
    
    def U_2_np(self, z):
        if self.scaling:
            return self.K * np.power(np.maximum(z, 1e-8), self.p)
        else:
            return self.K * np.log(np.maximum(z, 1e-8) + 1)
    
    def dU_2_np(self, z):
        if self.scaling:
            return self.p * self.K * np.power(np.maximum(z, 1e-8), self.p - 1)
        else:
            return self.K / (np.maximum(z, 1e-8) + 1)
    
    def utility_np(self, x, R):
        return np.where(x > R,
                            self.U_1_np(x - R),
                            - self.U_2_np(R - x)
                            )

    def utilityx_np(self, x, R):
        return np.where(x > R,
                            self.dU_1_np(x - R),
                            self.dU_2_np(R - x)
                            )
    
    

    def k_np(self, lam, L, R):
        return (self.utility_np(L, R) + lam  + self.U_2_np(R)) / np.maximum(L, 1e-1)
    
    def x_hat_np(self, R):
        return np.where(R > 0,
            R + (np.sqrt(self.K ** 2 * np.power(R, 2 * self.p) + R) - self.U_2_np(R)) ** 2,
            0.0 * R
            )
            
    def L_0_np(self, lam, L, R):
        return np.where(
            L >= self.x_hat_np(R),
            np.where(
                self.k_np(lam, L, R) > self.c_np(self.x_hat_np(R), R),
                R * 0.0,
                (
                    R
                    + quadratic(
                        0.5,
                        -(np.sqrt(np.maximum(L - R, 1e-4)) + lam),
                        0.5 * (L - R)
                        ) ** 2
                    )
            ),
            R * 0.0
            )
                    
    def z_0_np(self, lam, L, R):
        return np.where(
            L >= R,
            np.where(
                self.k_np(lam, L, R) > self.c_np(L, R),
                R * 0.0,
                (
                    R
                    + quadratic(
                        0.5,
                        self.U_2_np(R) + lam,
                        - 0.5 * R
                        ) ** 2
                    )
                ),
             lam * 0.0
             )

            
    def z_tilde_np(self, L, R):
        return np.where(
            L >= R,
            R * 0.0,
            (
                R
                + quadratic(
                    0.5,
                    self.U_2_np(np.maximum(R - L, 1e-4)),
                    0.5 * (L - R)
                    ) ** 2
                )
            )
        
    def z_tilde_0_np(self, lam, L, R):
        return np.where(
            L >= R,
            R * 0.0,
            np.where( 
               self.k_np(lam, L, R) > self.c_np(self.z_tilde_np(L, R), R),
               R * 0.0,
               (
                    R
                    + quadratic(
                        0.5,
                        self.U_2_np(R) + lam,
                        - 0.5 * R
                        ) ** 2
                    )
               )
            )

    def u_nonconcave_np(self, x, lam, L, R):
        return (
            self.utility(x, R)
            + lam * np.where(x > L, np.ones_like(x), np.zeros_like(x))
            )    
    
    def u_nonconcavex_np(self, x, R):
        return (
            self.utilityx(x, R)
            ) 
                   
    def u_np(self, x, lam, L, R):
        return np.where(
            L >= self.x_hat_np(R),
            np.where( 
                self.k_np(lam, L, R) > self.c_np(self.x_hat_np(R), R),
                ( 
                    np.where(x >= L,
                                self.U_1_np(x-R) + lam, 
                                self.k_np(lam, L, R) * x - self.U_2_np(R)
                                )
                    ),
                ( 
                    np.where(x >= L,
                                self.U_1_np(x-R) + lam, 
                                np.where(x >= self.L_0_np(lam, L, R),
                                         self.c_np(self.L_0_np(lam, L, R), R) * (x - L) + self.U_1_np(L - R) + lam,
                                         np.where(x >= self.x_hat_np(R),
                                                  self.U_1_np(x - R),
                                                  self.c_np(self.x_hat_np(R), R) * x - self.U_2_np(R)
                                                  )
                                         )
                                )
                    )
                ),
            np.where (
                L >= R,
                np.where( 
                    self.k_np(lam, L, R) >= self.c_np(L, R),
                    ( 
                        np.where(x >= L,
                                    self.U_1_np(x-R) + lam, 
                                    self.k_np(lam, L, R) * x - self.U_2_np(R)
                                    )    
                        ),
                    ( 
                        np.where(x >= self.z_0_np(lam, L, R),
                                    self.U_1_np(x - R) + lam,
                                    self.c_np(self.z_0_np(lam, L, R), R) * x - self.U_2_np(R)
                                    )
                        )
                ),
                np.where(
                    self.k_np(lam, L, R) > self.c_np(self.z_tilde_np(L, R), R),
                    ( 
                        np.where(x >= self.z_tilde_np(L, R),
                                    self.U_1_np(x - R) + lam,
                                    np.where(x >= L,
                                             self.c_np(self.z_tilde_np(L, R), R) * (x - L) - self.U_2_np(R - L) + lam,
                                             self.k_np(lam, L, R) * x - self.U_2_np(R)
                                             )
                                    )
                        ),
                    ( 
                        np.where(x >= self.z_tilde_0_np(lam, L, R),
                                    self.U_1_np(x - R) + lam,
                                    self.c_np(self.z_tilde_0_np(lam, L, R), R) * x - self.U_2_np(R)
                                    )
                        )
                    )
                )
            )
    
    def ux_np(self, x, lam, L, R):
        epsilon = 1e-3
        return np.where(
            L >= self.x_hat_np(R),
            np.where( 
                self.k_np(lam, L, R) > self.c_np(self.x_hat_np(R), R),
                ( 
                    np.where(x >= L,
                                self.dU_1_np(x-R), 
                                self.k_np(lam, L, R) + epsilon * (L - x)
                                )
                    ),
                ( 
                    np.where(x >= L,
                                self.dU_1_np(x-R), 
                                np.where(x >= self.L_0_np(lam, L, R),
                                         self.c_np(self.L_0_np(lam, L, R), R),
                                         np.where(x >= self.x_hat_np(R),
                                                  self.dU_1_np(x - R),
                                                  self.c_np(self.x_hat_np(R), R) + epsilon * (self.x_hat_np(R) - x)
                                                  )
                                         )
                                )
                    )
                ),
            np.where (
                L >= R,
                np.where( 
                    self.k_np(lam, L, R) >= self.c_np(L, R),
                    ( 
                        np.where(x >= L,
                                    self.dU_1_np(x-R), 
                                    self.k_np(lam, L, R) - epsilon * x
                                    )    
                        ),
                    ( 
                        np.where(x >= self.z_0_np(lam, L, R),
                                    self.dU_1_np(x - R),
                                    self.c_np(self.z_0_np(lam, L, R), R) + epsilon * (self.z_0_np(lam, L, R) - x)
                                    )
                        )
                ),
                np.where(
                    self.k_np(lam, L, R) > self.c_np(self.z_tilde_np(L, R), R),
                    ( 
                        np.where(x >= self.z_tilde_np(L, R),
                                    self.dU_1_np(x - R),
                                    np.where(x >= L,
                                             self.c_np(self.z_tilde_np(L, R), R),
                                             self.k_np(lam, L, R)  + epsilon * (L - x)
                                             )
                                    )
                        ),
                    ( 
                        np.where(x >= self.z_tilde_0_np(lam, L, R),
                                    self.dU_1_np(x - R),
                                    self.c_np(self.z_tilde_0_np(lam, L, R), R) + epsilon * (self.z_tilde_0_np(lam, L, R) - x)
                                    )
                        )
                    )
                )
            )
            
    def I_np(self, y):
        return np.power(y / self.p, 1 / (self.p - 1))
            
            
    def u_tilde_np(self, y, lam, L, R):
        return self.u_np(self.x(y, lam, L, R), lam, L, R) - y * self.x_np(y, lam, L, R)
    
    def u_tildey_np(self, y, lam, L, R):
        return - self.x(y, L, R)

    
    def x_np(self, y, lam, L, R):
        if L >= self.x_hat_np(R):
            if self.k_np(lam, L, R) > self.c_np(self.x_hat_np(R), R):
                return np.where(y >= self.k_np(lam, L, R),
                                np.zeros_like(y),
                                np.where(y >= self.c_np(L, R),
                                         L * np.ones_like(y),
                                         R + self.I(y)                                  
                                         )
                                )
            else:
                return np.where(y >= self.c_np(self.x_hat_np(R), R),
                                np.zeros_like(y),
                                np.where(y >= self.c_np(self.L_0_np(lam, L, R), R),
                                         R + self.I(y),
                                         np.where(y >= self.c_np(L, R),
                                                  L * np.ones_like(y),
                                                  R + self.I(y)
                                                  )
                                         )
                                )
            
        elif L >= R:
            if self.k_np(lam, L, R) >= self.c_np(L, R):
                return np.where(y >= self.k_np(lam, L, R),
                                np.zeros_like(y),
                                np.where(y >= self.c_np(L, R),
                                         L * np.ones_like(y),
                                         R + self.I(y)                                  
                                         )
                                ) 
            else:
                return np.where(y >= self.c_np(self.z_0_np(lam, L, R), R),
                                np.zeros_like(y),
                                R + self.I(y)
                                )
        else:
            if self.k_np(lam, L, R) > self.c_np(self.z_tilde_np(L, R), R): 
                return np.where(y >= self.k_np(lam, L, R),
                                np.zeros_like(y),
                                np.where(y >= self.c_np(self.z_tilde_np(L, R), R),
                                         L * np.ones_like(y),
                                         R + self.I(y)
                                         )
                                )
            else:
                return np.where(y >= self.c_np(self.z_tilde_0_np(lam, L, R), R),
                                np.zeros_like(y),
                                R + self.I(y)
                                ) 
            
            
    def H(self, Gx, Gxx, x, pi):
        return (
            (
                x * (self.alpha - self.a - (self.p - 1) * self.b ** 2)
                + pi * self.sigma * ( self.theta  + self.rho * self.b * (self.p - 1))
                ) * Gx
            + 0.5 * (tf.square(pi) * self.sigma ** 2 
                     - 2 * pi * x * self.sigma * self.rho * self.b
                     + tf.square(x) * self.b ** 2
                     ) * Gxx
            )
    
    
    def L(self, Gt, Gx, Gxx, x, pi):
        return Gt + self.H(Gx, Gxx, x, pi)
    
    def control_denominator(self, Mx, Mxx, x):
        return Mxx * self.sigma + 1e-8
            
    def control_numerator(self, Mx, Mxx, x):
        return  (
            (self.theta + self.rho * self.b * (self.p - 1)) * Mx
            - x * self.rho * self.b  * Mxx
        )
    
    
    def H_r(self, Gx, Gr, Gxx, Gxr, Grr, x, r, pi):
        return (
            (
                x * self.alpha 
                + pi * self.sigma * self.theta
                ) * Gx
            + 0.5 * (
                tf.square(pi) * self.sigma ** 2 
                - 2 * pi * x * self.sigma * self.rho * self.b
                + tf.square(x) * self.b ** 2
                ) * Gxx
            + self.a * r * Gr
            + 0.5 * tf.square(r) * self.b ** 2 * Grr
            + self.rho * self.sigma * pi * self.b * r * Gxr
        )
    
    
    def L_r(self, Gt, Gx, Gr, Gxx, Gxr, Grr, x, r, pi):
        return Gt + self.H_r(Gx, Gr, Gxx, Gxr, Grr, x, r, pi)
    
    def control_denominator_r(self, Gx, Gr, Gxx, Gxr, Grr, x, r):
        return Gxx * self.sigma + 1e-8
            
    def control_numerator_r(self, Gx, Gr, Gxx, Gxr, Grr, x, r):
        return  (
            self.theta * Gx
            + self.rho * self.b * r * Gxr
        )


    
    def H_dual(self, Gy, Gyy, y):
        ret = (
            - y * Gy * self.alpha_bar
            - 0.5 * (1 - self.rho ** 2) * tf.square(Gy * self.b) /  Gyy
            )
        
        if self.pi_max < 100:
        
            A = Gyy * (1 / self.sigma) ** 2 
            B = - y * self.b * self.rho * Gy / self.sigma + Gyy * self.theta_bar * y ** 2 / self.sigma
            
            v_pos = tf.minimum(0.0 * y, - B / A)
            v_neg = tf.maximum(0.0 * y, (self.pi_max - B) / A)
            
            term_pos = 0.5 * A * v_pos ** 2 + B * v_pos
            term_neg = 0.5 * A * v_neg ** 2 + (B - self.pi_max) * v_neg
            
            ret += (
                - y * Gy * self.b * self.rho * self.theta_bar
                + tf.minimum(term_pos, term_neg)
                )
                
        else:
            ret += 0.5 * tf.square(y) * Gyy * self.theta_bar ** 2
            
        return ret

    
    
    def L_dual(self, Gt, Gy, Gyy, y):
        return Gt + self.H_dual(Gy, Gyy, y) 
    

    def H_dual_r(self, Gy, Gr, Gyy, Gyr, Grr, y, r):
        ret = (
            - y * Gy * self.alpha_bar
            + 0.5 * tf.square(y) * Gyy * self.theta ** 2
            + self.a * r * Gr
            + 0.5 * self.b ** 2 * tf.square(r) * Grr
            - r * y * self.b * self.rho * self.theta * Gyr
            - 0.5 * (1 - self.rho ** 2) * tf.square(r * Gyr * self.b) /  Gyy
            )
            
        return ret

    
    
    def L_dual_r(self, Gt, Gy, Gr, Gyy, Gyr, Grr, y, r):
        return Gt + self.H_dual_r(Gy, Gr, Gyy, Gyr, Grr, y, r) 
    

class Solution(object):
    def __init__(self, config):
        self.config = config   
        self.method = Method(config)
        
        self.x_hat = config.x_hat_np(config.R)
        
        xaxis = np.linspace(config.x_range[0], config.x_range[1], 20)

        
        
        
        if config.lam * config.lower > 0:
            #solution via monte carlo simulation
            yaxis = np.linspace(config.y_range[0], config.y_range[1], solution_size)
            xaxis = np.linspace(config.x_range[0], config.x_range[1], solution_size)
            
            Y = ( 
                np.ones(shape = [solution_size, monte_size])
                * yaxis[:, np.newaxis]
                * np.exp(-(config.alpha_bar + 0.5 * config.theta_bar ** 2) * config.T - config.theta * np.sqrt(config.T) * np.random.normal(size = [solution_size, monte_size]))
                )
            
            self.states = xaxis
            vs = np.mean(config.u_tilde_np(Y, config.lam, config.lower, config.R), axis = 1)
            
            self.values = np.min(vs + xaxis[:,np.newaxis] * yaxis, axis = 1)
            self.yaxis = yaxis[np.argmin(vs + xaxis[:,np.newaxis] * yaxis, axis = 1)]


            

            
    def value_function(self, t, x):
        return np.exp((self.config.alpha + self.config.theta ** 2 / (2 * (1 - self.config.p))) * self.config.p * (self.config.T - t)) * np.power(x, self.config.p)
    
    def value_jacobian(self, t, x):
        return self.config.p * np.exp((self.config.alpha + self.config.theta ** 2 / (2 * (1 - self.config.p))) * self.config.p * (self.config.T - t)) * np.power(x, self.config.p - 1)
    
    def value_hessian(self, t, x):
        return self.config.p * (self.config.p - 1) * np.exp((self.config.alpha + self.config.theta ** 2 / (2 * (1 - self.config.p))) * self.config.p * (self.config.T - t)) * np.power(x, self.config.p - 2)
    
    def value_time(self, t, x):
        return - (self.config.alpha + self.config.theta ** 2 / (2 * (1 - self.config.p))) * self.config.p * np.exp((self.config.alpha + self.config.theta ** 2 / (2 * (1 - self.config.p))) * self.config.p * (self.config.T - t)) * np.power(x, self.config.p)
        

    def u_np(self, x):
        return np.where(x >= self.config.R,
                        np.power(x-self.config.R + 1e-8, self.config.p), 
                        - self.config.K * np.power(self.config.R - x + 1e-8, self.config.p) 
                        )
                       
    def u_np_conv(self, x):
        return np.where(x >= self.x_hat,
                        np.power(x-self.config.R + 1e-8, self.config.p),
                        x * self.config.p * np.power(self.x_hat - self.config.R + 1e-8, self.config.p - 1)  - self.config.K * np.power(self.config.R, self.config.p)
                        )
    
    
    def func2(self, x, y):
        return -self.u_np_conv(x) + x * y
    
    def tilde_u(self, y):
        return -minimize(lambda x : self.func2(x, y), y, bounds = [(0, np.inf)]).fun
 
    def tilde_u_np(self, y):
        return (
        (np.power(y/self.config.p, self.config.p / (self.config.p - 1)) * (1 - self.config.p) - self.config.R * y) * (y < self.config.c_np(self.x_hat, self.config.R))
        - self.config.U_2_np(self.config.R) * (y > self.config.c_np(self.x_hat, self.config.R))
        )
    
    
    
    def k(self, tau, y):
        return (np.log(y) - np.log(self.config.c_np( self.x_hat, self.config.R)) - (self.config.alpha_bar + 0.5 * self.config.theta_bar ** 2) * tau ) / (self.config.theta_bar * np.sqrt(tau) + 1e-8)
    
    
    def v(self, t, y):
        tau = self.config.T - t
        
        term1 = (- self.config.U_2_np(self.config.R)) * cdf(self.k(tau, y))
       
        term2 = ( 
            np.power(y / self.config.p, self.config.p / (self.config.p - 1))
            * (1 - self.config.p)
            * np.exp(- (self.config.p / (self.config.p - 1)) * (self.config.alpha_bar - self.config.theta_bar ** 2 / (2 * (self.config.p - 1))) * tau)
            * cdf(-self.k(tau, y) - self.config.p * self.config.theta_bar * np.sqrt(tau) / (self.config.p - 1))
            )
        
        term3 = self.config.R * y * np.exp(-self.config.alpha_bar * tau) * cdf(-self.k(tau, y) - self.config.theta_bar * np.sqrt(tau))
        
        return term1 + term2 - term3
    
    
    
    def d_v(self, t, y):
        tau = self.config.T - t
       
        term1 = ( 
            np.power(y / self.config.p, 1 / (self.config.p - 1))
            * np.exp(- (self.config.p / (self.config.p - 1)) * (self.config.alpha_bar - self.config.theta_bar ** 2 / (2 * (self.config.p - 1))) * tau)
            * cdf(-self.k(tau, y) - self.config.p * self.config.theta_bar * np.sqrt(tau) / (self.config.p - 1))
            )
        
        term2 = self.config.R * np.exp(-self.config.alpha_bar * tau) * cdf(-self.k(tau, y) - self.config.theta_bar * np.sqrt(tau))
        
        return - (term1 + term2)
    
    
    def d2_v(self, t, y):
        tau = self.config.T - t
        term1 = - ( 
            np.power(y, (2 - self.config.p) / (self.config.p - 1))
            * np.power(self.config.p, 1 / (1 - self.config.p)) / (self.config.p - 1)
            * np.exp(- (self.config.p / (self.config.p - 1)) * (self.config.alpha_bar - self.config.theta_bar ** 2 / (2 * (self.config.p - 1))) * tau)
            * cdf(-self.k(tau, y) - self.config.p * self.config.theta_bar * np.sqrt(tau) / (self.config.p - 1))
            )
        term2 = self.x_hat * self.config.c_np( self.x_hat, self.config.R) * pdf(self.k(tau, y)) / (self.config.theta_bar * np.sqrt(tau) * y ** 2)
        
        return term1 + term2
    
    def pi(self, t, y):
        return - (self.config.theta_bar / self.config.sigma) * self.d2_v(t, y) * y / self.d_v(t, y) + self.config.rho * self.config.b / self.config.sigma
    
    
    def control(self, t, x):
        y = self.dual_state(t, x)
        return self.pi(t, y)
    
    def dual_state(self, t, x):
        t = min(t, self.config.T * 0.99)
        if self.config.R <= 0.01:
            tau = self.config.T - t
           
            term1 = ( 
                np.exp(- self.config.p * (self.config.alpha_bar - self.config.theta_bar ** 2 / (2 * (self.config.p - 1))) * tau)
                )
            return self.config.p * np.power(x, self.config.p - 1) / term1
        else:
            ys = np.linspace(self.config.y_range[0], self.config.y_range[1], sample_size)
            return ys[np.argmin(self.v(t, ys) + x * ys)]

                
                           
    
    def func3(self, t, x, y):
        return self.v(t, y) + x * y
    
    def value(self, t, x):
        try:
            while len(x) > 0:
                x = x[0]
        except:
            pass
        
        t = min(t, self.config.T  * 0.99)
        dual = self.dual_state(t, x)
        return self.v(t, dual) + x * dual + self.config.lam
       















class FeedForwardSubNet(tf.keras.Model):
    def __init__(self, dim = 1, positive = False, negative = False, fn = None):
        assert not (positive and negative)
        super(FeedForwardSubNet, self).__init__()
        self.positive = positive
        self.negative = negative
        self.fn = fn        
        
        self.dense_layers = [tf.keras.layers.Dense(num_hiddens[i],
                                                   use_bias=True,
                                                   activation=None)
                             for i in range(len(num_hiddens))]
        #final output should be gradient of size dim
        self.dense_layers.append(tf.keras.layers.Dense(dim, 
                                                       use_bias=True, 
                                                       activation=None))

    def call(self, x, training):
        for i in range(len(self.dense_layers) - 1):
            x = self.dense_layers[i](x)
            if self.fn is not None:
                x = self.fn(x)
            else:
                x =  tf.tanh(x)
        x = self.dense_layers[-1](x)
        if self.positive:
            x = tf.nn.softplus(x) + 1e-4
        if self.negative:
            x = -tf.nn.softplus(x) - 1e-4
        return x 
    
    

def penalise_range(x, lower = -np.inf, upper = np.inf):
    return  10 * tf.square(tf.where(
        tf.logical_or(x > upper, x < lower),
        tf.maximum(x - upper, lower - x),
        0 
        ))


def evaluate(pi, t, x, config, method):
    start = time.time()
    x_arr = np.split(x, method.bsde_N + 1)
    pi_arr = np.split(pi, method.bsde_N + 1)
    ind_arr = [np.argsort(x) for x in x_arr]
    pis = [lambda x: np.interp(x, x_arr[i][ind_arr[i]], pi_arr[i][ind_arr[i]]) for i in range(method.bsde_N + 1)]
    
    pi = lambda t, x: pis[int(t * (method.bsde_N + 1) / config.T)](x)
    
    
    xaxis = np.linspace(config.x_range[0], config.x_range[1], monte_M)
    x = np.ones(shape = [monte_M, monte_size]) * xaxis[:, np.newaxis]
    
    N = monte_N
    dt = config.T / N
    
    dw = np.random.normal(size = [monte_M, monte_size, N]) * np.sqrt(dt)
    W = np.sum(dw, axis = 2)
    
    
    for i in range(N):
        x = (
            x
            + x  * config.alpha * dt
            + pi(i * dt, x) * config.sigma * config.theta * dt
            + pi(i * dt, x)  * config.sigma * dw[:, :, i]
            )
        
        
    b = np.sign(config.rho) * config.b
    R = config.R * np.exp((config.a - 0.5 * b ** 2) * config.T + b * W)
    L = config.lower * np.exp((config.La - 0.5 * config.Lb ** 2) * config.T  + config.Lb * W)
    
    log(f'Monte Carlo runtime: {time.time()-start:.2f}')
    
    return xaxis, x, np.mean(config.u_nonconcave_np(x, config.lam, L, R), axis = 1), L, R
                

    
def plot(config, method, data, processes, activation_function):
    
    #solution
    solution = Solution(config)
    xaxis = np.linspace(config.x_range[0], config.x_range[1], sample_size)
    yaxis = np.linspace(config.y_range[0], config.y_range[1], sample_size)

    mid = (config.x_range[1] + config.x_range[0]) / 2
    length = config.x_range[1] - config.x_range[0]
    yaxis0 = yaxis[np.abs(-solution.d_v(test_time, yaxis) - mid) < length / 2]

    if config.lam * config.lower == 0:
        states = -solution.d_v(test_time, yaxis0)
        controls = states * solution.pi(test_time, yaxis0)
        values = [solution.value(test_time, x) for x in xaxis]
        xaxis0 = xaxis
        derivs2 = solution.d2_v(test_time, yaxis0)
    else:
        values = solution.values
        states = solution.states
        xaxis0 = solution.states
        yaxis0 = solution.yaxis
    
    
    if len(data['M losses']) > 0 and activation_function != 'early' and type(activation_function) == str:
        if 'control losses' in data.keys():
            fig, axes = plt.subplots(1)
            axes.plot(data['control losses'])
            axes.set_title('Control Losses')
            axes.set_yscale('log')
            axes.set_xlabel('Iteration Step')
            axes.set_ylabel('Loss')
            axes.grid(0.5)
        fig, axes = plt.subplots(1)
        axes.plot(data['M losses'])
        axes.set_title(method.upper() + ' Loss')
        axes.set_yscale('log')
        axes.set_xlabel('Iteration Step')
        axes.set_ylabel('Loss')
        axes.grid(0.5)    
        
    t0   = processes['t'  ].numpy()[:,0]
    x0   = processes['x'  ].numpy()[:,0]
    
    mask = True#np.isclose(t0, test_time)# & (x0 >= config.x_range[0]) & (x0 <= config.x_range[1])
    
    t   = t0[mask]
    x   = x0[mask]
    pi  = processes['pi' ].numpy()[:,0][mask]
    if 'M' in processes.keys():
        M   = processes['M'  ].numpy()[:,0][mask]
    if 'Mt' in processes.keys():
        Mt  = processes['Mt' ].numpy()[:,0][mask]
    Mx  = processes['dM' ].numpy()[:,0][mask]
    if 'dM2' in processes.keys():
        Mxx = processes['dM2'].numpy()[:,0][mask]
        
    if method == 'smp':
        pi0  = processes['pi' ].numpy()[:, 0]
        monteX, monteXT, monteM, L, R = evaluate(pi0, t0, x0, config, Method(config))
        
        
        plt.figure()
    
    
        plt.axvline(config.R, label = '$R$', linestyle = 'dotted', c = 'C3')
        plt.axvline(config.lower, label = '$L$', linestyle = 'dotted', c = 'C4')
        
        if config.lower >= config.x_hat_np(config.R):
            if config.k_np(config.lam, config.lower, config.R) > config.c_np(config.x_hat_np(config.R), config.R):
                pass
            else:
                plt.axvline(config.L_0_np(config.lam, config.lower, config.R), label = '$L_0$', linestyle = 'dotted', c = 'C6')
                plt.axvline(config.x_hat_np(config.R), label = '$z$', linestyle = 'dotted', c = 'C5')
        elif config.lower >= config.R:
            if config.k_np(config.lam, config.lower, config.R) >= config.c_np(config.lower, config.R):
                pass
            else:
                plt.axvline(config.z_0_np(config.lam, config.lower, config.R), label = '$z_0$', linestyle = 'dotted', c = 'C6')
        else:
            if config.k_np(config.lam, config.lower, config.R) > config.c_np(config.z_tilde_np(config.lower, config.R), config.R):
                plt.axvline(config.z_tilde_np(config.lower, config.R), label = '$\\tilde{z}$', linestyle = 'dotted', c = 'C6')
            else:
                plt.axvline(config.z_tilde_0_np(config.lam, config.lower, config.R), label = '$\\tilde{z}_0$', linestyle = 'dotted', c = 'C7')
    
        plt.grid(0.5)
        plt.xlabel('$X_T$')
        plt.ylabel('count')
        plt.title('Distribution of $X_T$')
        plt.legend()
        plt.xlim(-1e-1, config.x_range[1])
        for xT in monteXT[::10]:
            h, edges = np.histogram(xT, bins = 50) 
            plt.stairs(h, edges)
        plt.show()
    
    
    
    
        plt.figure()
    
    
        plt.axvline(config.R, label = '$R$', linestyle = 'dotted', c = 'C3')
        plt.axvline(config.lower, label = '$L$', linestyle = 'dotted', c = 'C4')
        
        if config.lower >= config.x_hat_np(config.R):
            if config.k_np(config.lam, config.lower, config.R) > config.c_np(config.x_hat_np(config.R), config.R):
                pass
            else:
                plt.axvline(config.L_0_np(config.lam, config.lower, config.R), label = '$L_0$', linestyle = 'dotted', c = 'C6')
                plt.axvline(config.x_hat_np(config.R), label = '$z$', linestyle = 'dotted', c = 'C5')
        elif config.lower >= config.R:
            if config.k_np(config.lam, config.lower, config.R) >= config.c_np(config.lower, config.R):
                pass
            else:
                plt.axvline(config.z_0_np(config.lam, config.lower, config.R), label = '$z_0$', linestyle = 'dotted', c = 'C6')
        else:
            if config.k_np(config.lam, config.lower, config.R) > config.c_np(config.z_tilde_np(config.lower, config.R), config.R):
                plt.axvline(config.z_tilde_np(config.lower, config.R), label = '$\\tilde{z}$', linestyle = 'dotted', c = 'C6')
            else:
                plt.axvline(config.z_tilde_0_np(config.lam, config.lower, config.R), label = '$\\tilde{z}_0$', linestyle = 'dotted', c = 'C7')
    
        plt.grid(0.5)
        plt.xlabel('$X_T$')
        plt.ylabel('$|U(x) - \\bar{U}(x)|$')
        plt.title('Concavification Gap')
        plt.legend()
        plt.xlim(-1e-1, config.x_range[1])
        for xT, LT, RT in zip(monteXT[::10], L[::10], R[::10]):
            plt.scatter( 
                xT,
                np.abs(config.u_np(xT, config.lam, LT, RT) - config.u_nonconcave_np(xT, config.lam, LT, RT))
            )
    
        plt.show()
        
        
        plt.figure()
    
    
        plt.axvline(config.R, label = '$R$', linestyle = 'dotted', c = 'C3')
        plt.axvline(config.lower, label = '$L$', linestyle = 'dotted', c = 'C4')
        
        if config.lower >= config.x_hat_np(config.R):
            if config.k_np(config.lam, config.lower, config.R) > config.c_np(config.x_hat_np(config.R), config.R):
                pass
            else:
                plt.axvline(config.L_0_np(config.lam, config.lower, config.R), label = '$L_0$', linestyle = 'dotted', c = 'C6')
                plt.axvline(config.x_hat_np(config.R), label = '$z$', linestyle = 'dotted', c = 'C5')
        elif config.lower >= config.R:
            if config.k_np(config.lam, config.lower, config.R) >= config.c_np(config.lower, config.R):
                pass
            else:
                plt.axvline(config.z_0_np(config.lam, config.lower, config.R), label = '$z_0$', linestyle = 'dotted', c = 'C6')
        else:
            if config.k_np(config.lam, config.lower, config.R) > config.c_np(config.z_tilde_np(config.lower, config.R), config.R):
                plt.axvline(config.z_tilde_np(config.lower, config.R), label = '$\\tilde{z}$', linestyle = 'dotted', c = 'C6')
            else:
                plt.axvline(config.z_tilde_0_np(config.lam, config.lower, config.R), label = '$\\tilde{z}_0$', linestyle = 'dotted', c = 'C7')
    
        plt.grid(0.5)
        plt.xlabel('$X_T$')
        plt.ylabel('$E[|U(x) - \\bar{U}(x)|^2]$')
        plt.title('Average Concavification Error')
        plt.legend()
        plt.xlim(-1e-1, config.x_range[1])
        for xT, LT, RT, x0 in zip(monteXT, L, R, monteX):
            plt.scatter( 
                x0,
                np.mean(
                    np.power(
                        config.u_np(xT, config.lam, LT, RT) - config.u_nonconcave_np(xT, config.lam, LT, RT),
                        2
                        )
                    ),
                c = 'C0',
                s = 1
            )
    
        plt.show()
        
    
    
    
    

            
    fig, axes = plt.subplots(2, 2, figsize = (12, 12))
    fig.suptitle(f'{method.upper()} Method, $t=${test_time}, {activation_function}', fontsize=30)
    
    axes[0,0].grid(0.5)
    axes[0,0].scatter(x[t < config.T - 1e-6],pi[t < config.T - 1e-6], c = t[t < config.T - 1e-6], s = 1)
    if config.lam * config.lower == 0:
        axes[0,0].plot(states, controls, label = 'Reference', c = 'r', linestyle = 'dashed')
    axes[0,0].axvline(config.x_hat_np(config.R), linestyle = 'dotted', c = 'm', label = '$\hat{z}$')
    axes[0,0].set_xlabel('$z$')
    axes[0,0].set_ylabel('$\pi(t, z)$')
    axes[0,0].set_title('control vs z')
    axes[0,1].grid(0.5)
    if 'M' in processes.keys():
        axes[0,1].scatter(x,M, c = t, s = 1)
    else:
        axes[0,1].plot(monteX,monteM, c = 'navy')
    axes[0,1].set_title('$\\bar{g}$ vs z')    
    axes[0,1].plot(xaxis0, values, label = 'Reference', c = 'r', linestyle = 'dashed')
    axes[0,1].set_xlabel('$z$')
    axes[0,1].set_ylabel('$\\bar{g}(t, z)$')
    axes[0,1].axvline(config.x_hat_np(config.R), linestyle = 'dotted', c = 'm', label = '$\hat{z}$')
    axes[1,0].grid(0.5)
    axes[1,0].scatter(x,Mx, c = t, s = 1)
    axes[1,0].set_title('$\partial_z\\bar{g}$ vs z')  
    axes[1,0].plot(states, yaxis0, label = 'Reference', c = 'r', linestyle = 'dashed')
    axes[1,0].set_xlabel('$z$')
    axes[1,0].set_ylabel('$\partial_z\\bar{g}$')
    axes[1,0].axvline(config.x_hat_np(config.R), linestyle = 'dotted', c = 'm', label = '$\hat{z}$')
    if 'dM2' in processes.keys():
        axes[1,1].grid(0.5)
        mask2 = (t < config.T - 1e-6) & (Mxx > threshold)
        axes[1,1].scatter(x[mask2],Mxx[mask2], c = t[mask2], s = 1)
        axes[1,1].set_title('$\partial_{zz}\\bar{g}$ vs z')      
        if config.lam * config.lower == 0:
            axes[1,1].plot(states, -1 / derivs2, label = 'Reference', c = 'r', linestyle = 'dashed')
        axes[1,1].set_xlabel('$z$')
        axes[1,1].set_ylabel('$\\bar{g}_{zz}(t, z)$')
        axes[1,1].axvline(config.x_hat_np(config.R), linestyle = 'dotted', c = 'm', label = '$\hat{z}$')
            
    

    plt.show()    
    


    
    
    

def make_plots():

    config = Problem()
    solution = Solution(config)
    
    
    
    
    xaxis = np.linspace(config.x_range[0], config.x_range[1], sample_size)
    yaxis = np.linspace(config.y_range[0], config.y_range[1], sample_size)
    mid = (config.x_range[1] + config.x_range[0]) / 2
    length = config.x_range[1] - config.x_range[0]
    
    
    plt.plot(xaxis, solution.u_np(xaxis), label = '$U$')
    plt.plot(xaxis, solution.u_np_conv(xaxis), linestyle = 'dashed', label = '$\\bar{U}$')
    plt.axvline(config.x_hat_np(config.R), linestyle = 'dotted', c = 'k', label = '$\hat{z}$')
    plt.grid(0.5)
    plt.xlabel('$z$')
    plt.ylabel('$U$')
    plt.title('Utility Function')
    plt.legend()
    plt.show()
    
    
    plt.figure()
    
    plt.plot(yaxis, [solution.tilde_u(y) for y in yaxis], label = '$\\bar{U}$')
    plt.axvline(config.c_np(config.x_hat_np(config.R), config.R), linestyle = 'dotted', c = 'k', label = '$\hat{u}$')
    plt.axhline(-config.U_2_np(config.R), linestyle = 'dotted', c = 'm', label = '$u_0$')
    # plt.plot(yaxis,solution.tilde_u_np(yaxis), linestyle = 'dotted' )
    plt.grid(0.5)
    plt.title('Dual Utility Function')
    plt.xlabel('$y$')
    plt.ylabel('$\\tilde{U}$')
    plt.legend()
    plt.show()
    
    
        
    
    
    
    
    
    
    
    
    
    
    plt.figure()
    for t in np.linspace(0, config.T, 3):
        plt.plot(yaxis, solution.v(t, yaxis), label = '$\\tilde{g}$' + f'({t:.1f}, y)')
    plt.axvline(config.c_np(config.x_hat_np(config.R), config.R), linestyle = 'dotted', c = 'k', label = '$\hat{u}$')
    plt.axhline(-config.U_2_np(config.R), linestyle = 'dotted', c = 'm', label = '$u_0$')
    plt.grid(0.5)
    plt.xlabel('$y$')
    plt.ylabel('$\\tilde{g}$')
    plt.title('Dual Value Fuction')
    plt.legend()
    plt.show()
    
    
    
    
    
    plt.figure()
    for t in np.linspace(0, config.T, 3):
        plt.plot(xaxis, [solution.value(t, x) for x in xaxis], label = '$\\bar{g}$' + f'({t:.1f}, z)')
    plt.plot(xaxis, solution.u_np_conv(xaxis), linestyle = 'dashed', label = '$\\bar{U}$')
    plt.plot(xaxis, solution.u_np(xaxis), linestyle = 'dashed', label = '$U$')
    # plt.plot(xaxis, solution.value_function(0, xaxis), linestyle = 'dotted', label = '$R=0$', c = 'r')
    
    plt.axvline(config.x_hat_np(config.R), linestyle = 'dotted', c = 'k', label = '$\hat{z}$')
    plt.grid(0.5)
    plt.xlabel('$z$')
    plt.ylabel('$\\bar{g}$')
    plt.title('Value Function')
    plt.legend()
    plt.show()
    


    
            
    plt.figure()
    for t in np.linspace(0, config.T, 3):
        plt.plot(yaxis, -solution.d_v(t, yaxis), label = '$-\partial_y \\tilde{g}$' + f'({t:.1f}, y)')
    plt.axvline(config.c_np(config.x_hat_np(config.R), config.R), linestyle = 'dotted', c = 'k', label = '$\hat{u}$')
    plt.axhline(config.x_hat_np(config.R), linestyle = 'dotted', c = 'm', label = '$\hat{z}$')
    plt.grid(0.5)
    plt.xlabel('$y$')
    plt.ylabel('$\partial_y \\tilde{g}$')
    plt.title('Derivative of Dual Value Function')
    plt.legend()
    plt.show()
    
    plt.figure()
    for t in np.linspace(0, config.T, 3):
        t = min(t, config.T * 0.99)
        plt.plot(yaxis, solution.pi(t, yaxis), label = '$\pi$' + f'({t:.1f}, y)')
    plt.axvline(config.c_np(config.x_hat_np(config.R), config.R), linestyle = 'dotted', c = 'k', label = '$\hat{u}$')
    plt.axhline(config.risk, linestyle = 'dotted', c = 'r', label = 'Risk Control')
    plt.grid(0.5)
    plt.xlabel('$y$')
    plt.ylabel('$\pi$')
    plt.legend()
    plt.title('Optimal Control (Proportion) vs Dual State')
    plt.yscale('log')
    plt.show()
    
    
    plt.figure()
    for t in np.linspace(0, config.T, 3):
        t = min(t, config.T * 0.99)
        plt.plot(yaxis, solution.pi(t, yaxis) * -solution.d_v(t, yaxis), label = '$z\pi$' + f'({t:.1f}, y)')
    plt.axvline(config.c_np(config.x_hat_np(config.R), config.R), linestyle = 'dotted', c = 'k', label = '$\hat{u}$')
    plt.grid(0.5)
    # plt.ylim(-2, 52)
    plt.xlabel('$y$')
    plt.title('Optimal Control (Amount) vs Dual State')
    
    plt.ylabel('$z\pi$')
    plt.legend()
    
    plt.show()
    
    plt.figure()
    for t in np.linspace(0, config.T, 3):
        t = min(t, config.T * 0.99)

        yaxis0 = yaxis[np.abs(-solution.d_v(t, yaxis) - mid) < length / 2]
        plt.plot(-solution.d_v(t, yaxis0), solution.pi(t, yaxis0), label = '$\pi$' + f'({t:.1f}, z)')
    plt.axhline(config.risk, linestyle = 'dotted', c = 'r', label = 'Risk Control')
    plt.axvline(config.x_hat_np(config.R), linestyle = 'dotted', c = 'm', label = '$\hat{z}$')
    plt.grid(0.5)
    plt.xlabel('$z$')
    plt.ylabel('$\pi$')
    plt.legend()
    plt.title('Optimal Control (Proportion) vs State')
    # plt.ylim(-0.5, 500)
    plt.yscale('log')
    plt.show()
    
    plt.figure()
    for t in np.linspace(0, config.T, 3):
        t = min(t, config.T * 0.99)

        yaxis0 = yaxis[np.abs(-solution.d_v(t, yaxis) - mid) < length / 2]
        plt.plot(-solution.d_v(t, yaxis0), -solution.d_v(t, yaxis0) * solution.pi(t, yaxis0), label = '$z\pi$' + f'({t:.1f}, z)')
    plt.axvline(config.x_hat_np(config.R), linestyle = 'dotted', c = 'm', label = '$\hat{z}$')
    plt.grid(0.5)
    plt.xlabel('$z$')
    plt.ylabel('$z\pi$')
    plt.legend()
    plt.title('Optimal Control (Amount) vs State')
    plt.show()
    
    plt.figure()
    for t in np.linspace(0, config.T, 3):
        t = min(t, config.T * 0.99)
        yaxis0 = yaxis[np.abs(-solution.d_v(t, yaxis) - mid) < length / 2]
        plt.plot(-solution.d_v(t, yaxis0), yaxis0, label = '$y$' + f'({t:.1f}, z)')
    
    # plt.plot(xaxis, solution.value_jacobian(0, xaxis), linestyle = 'dotted', label = '$R=0$', c = 'r')
    plt.axvline(config.x_hat_np(config.R), linestyle = 'dotted', c = 'm', label = '$\hat{z}$')
    plt.grid(0.5)
    plt.xlabel('$z$')
    plt.ylabel('$y$')
    plt.legend()
    plt.title('Dual State vs State')
    # plt.ylim(-0.1, 1.0)
    plt.show()
    
    # mask = solution.value_hessian(0, xaxis) > min(-1 / solution.d2_v(0, yaxis0))
    
    plt.figure()
    for t in np.linspace(0, config.T, 3):
        t = min(t, config.T * 0.99)
        yaxis0 = yaxis[(np.abs(-solution.d_v(t, yaxis) - mid) < length / 2) & (solution.d2_v(t, yaxis) > 0.01)]
        plt.plot(-solution.d_v(t, yaxis0), -1 / solution.d2_v(t, yaxis0), label = '$\\bar{g}_{zz}$' + f'({t:.1f}, z)')
    # plt.plot(xaxis[mask], solution.value_hessian(0, xaxis)[mask], linestyle = 'dotted', label = '$R = 0$', c = 'r')
    plt.axvline(config.x_hat_np(config.R), linestyle = 'dotted', c = 'm', label = '$\hat{z}$')
    plt.grid(0.5)
    plt.xlabel('$z$')
    plt.ylabel('$\\bar{g}_{zz}$')
    plt.legend()
    plt.title('2nd Derivative vs State')
    # plt.ylim(-0.4, 0.1)
    plt.show()
        
    
if __name__ == '__main__':
    make_plots()
    
    
    
    
        
        
        
        
