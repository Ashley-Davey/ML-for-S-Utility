# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 18:39:12 2024

@author: ashle
"""

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt, rcParams
from common import Problem, Solution, Method, evaluate, log
from PINN import PINNSolver
from PINN_general import PINN_generalSolver
# from BSDE import BSDESolver
from SMP import SMPSolver
from PINNDual import PINNDualSolver
from PINNDual_general import PINNDual_generalSolver

rcParams['figure.dpi'] = 600

def main():
    
    # rho = 1, concave
    # rho = 0, concave
    # rho = 1.0, nonconcave
    # rho = 0.0, nonconcave
    
    threshold = -10
    num_repeats = {
        'primal' : 1,
        'primal_nonconcave' : 1,
        'primal_general' : 1,
        'primal_general_nonconcave' : 1,
        'dual' : 1,
        'dual_general' : 1,
        'smp' : 1,
        'smp_nonconcave' : 1,
        }
    
    for rho in [1.0, 0.0]:
    # for rho, non_concave in [ [0.0, True]]:
    
        config = Problem(rho = rho, non_concave = False, scaling = True)
        config_nonconcave = Problem(rho = rho, non_concave = True, scaling = True)


        primal = config.scaling
        primal_nonconcave = config.scaling
        primal_general = not config.scaling
        primal_general_nonconcave = not config.scaling
        dual = config.scaling
        dual_general = not config.scaling
        smp = True
        smp_nonconcave = False
        soln = (rho == 1.0) and config.scaling
                
        
        method = Method(config)
        solution = Solution(config)
        if primal:
            repeats = num_repeats['primal']
            for n in range(repeats):
                log(f'Run {n}')
                tf.keras.backend.clear_session()
                tf.keras.backend.set_floatx('float64')
                PINNdata, PINNProcesses, PINNProcessesB = PINNSolver(config).train()
                
                t0   = PINNProcesses['t'  ].numpy()[:, 0]
                x0   = PINNProcesses['x'  ].numpy()[:, 0]
                
                PINNmask = (x0 >= config.x_range[0]) & (x0 <= config.x_range[1]) #& np.isclose(t0, method.test_time)
                    
                if n == 0:
                    PINNt   = PINNProcesses['t'  ].numpy()[:, 0][PINNmask] / repeats
                    PINNx   = PINNProcesses['x'  ].numpy()[:, 0][PINNmask] / repeats
                    PINNpi  = PINNProcesses['pi' ].numpy()[:, 0][PINNmask] / repeats
                    PINNM   = PINNProcesses['M'  ].numpy()[:, 0][PINNmask] * np.exp(config.discount * (config.T - PINNt)) / repeats
                    PINNMx  = PINNProcesses['dM' ].numpy()[:, 0][PINNmask] * np.exp(config.discount * (config.T - PINNt)) / repeats
                    PINNMxx = PINNProcesses['dM2'].numpy()[:, 0][PINNmask] * np.exp(config.discount * (config.T - PINNt)) / repeats
                else:
                    PINNt   += PINNProcesses['t'  ].numpy()[:, 0][PINNmask] / repeats
                    PINNx   += PINNProcesses['x'  ].numpy()[:, 0][PINNmask] / repeats
                    PINNpi  += PINNProcesses['pi' ].numpy()[:, 0][PINNmask] / repeats
                    PINNM   += PINNProcesses['M'  ].numpy()[:, 0][PINNmask] * np.exp(config.discount * (config.T - PINNt)) / repeats
                    PINNMx  += PINNProcesses['dM' ].numpy()[:, 0][PINNmask] * np.exp(config.discount * (config.T - PINNt)) / repeats
                    PINNMxx += PINNProcesses['dM2'].numpy()[:, 0][PINNmask] * np.exp(config.discount * (config.T - PINNt)) / repeats
                
                
        if primal_nonconcave:
            repeats = num_repeats['primal_nonconcave']
            for n in range(repeats):
                log(f'Run {n}')
                tf.keras.backend.clear_session()
                tf.keras.backend.set_floatx('float64')
                PINNdata, PINNProcesses, PINNProcessesB = PINNSolver(config_nonconcave).train()
                
                t0   = PINNProcesses['t'  ].numpy()[:, 0]
                x0   = PINNProcesses['x'  ].numpy()[:, 0]
                
                PINNmask = (x0 >= config.x_range[0]) & (x0 <= config.x_range[1]) #& np.isclose(t0, method.test_time)
                    
                if n == 0:
                    PINNt_nonconcave   = PINNProcesses['t'  ].numpy()[:, 0][PINNmask] / repeats
                    PINNx_nonconcave   = PINNProcesses['x'  ].numpy()[:, 0][PINNmask] / repeats
                    PINNpi_nonconcave  = PINNProcesses['pi' ].numpy()[:, 0][PINNmask] / repeats
                    PINNM_nonconcave   = PINNProcesses['M'  ].numpy()[:, 0][PINNmask] * np.exp(config.discount * (config.T - PINNt)) / repeats
                    PINNMx_nonconcave  = PINNProcesses['dM' ].numpy()[:, 0][PINNmask] * np.exp(config.discount * (config.T - PINNt)) / repeats
                    PINNMxx_nonconcave = PINNProcesses['dM2'].numpy()[:, 0][PINNmask] * np.exp(config.discount * (config.T - PINNt)) / repeats
                else:
                    PINNt_nonconcave   += PINNProcesses['t'  ].numpy()[:, 0][PINNmask] / repeats
                    PINNx_nonconcave   += PINNProcesses['x'  ].numpy()[:, 0][PINNmask] / repeats
                    PINNpi_nonconcave  += PINNProcesses['pi' ].numpy()[:, 0][PINNmask] / repeats
                    PINNM_nonconcave   += PINNProcesses['M'  ].numpy()[:, 0][PINNmask] * np.exp(config.discount * (config.T - PINNt)) / repeats
                    PINNMx_nonconcave  += PINNProcesses['dM' ].numpy()[:, 0][PINNmask] * np.exp(config.discount * (config.T - PINNt)) / repeats
                    PINNMxx_nonconcave += PINNProcesses['dM2'].numpy()[:, 0][PINNmask] * np.exp(config.discount * (config.T - PINNt)) / repeats
        
        
        if primal_general:
            repeats = num_repeats['primal_general']
            for n in range(repeats):
                log(f'Run {n}')
                tf.keras.backend.clear_session()
                tf.keras.backend.set_floatx('float64')
                PINNdata, PINNProcesses, PINNProcessesB = PINN_generalSolver(config).train()
                
                t0   = PINNProcesses['t'  ].numpy()[:, 0]
                x0   = PINNProcesses['x'  ].numpy()[:, 0]
                
                PINNmask = (x0 >= config.x_range[0]) & (x0 <= config.x_range[1]) #& np.isclose(t0, method.test_time)
                    
                if n == 0:
                    PINNt_general   = PINNProcesses['t'  ].numpy()[:, 0][PINNmask] / repeats
                    PINNx_general   = PINNProcesses['x'  ].numpy()[:, 0][PINNmask] / repeats
                    PINNpi_general  = PINNProcesses['pi' ].numpy()[:, 0][PINNmask] / repeats
                    PINNM_general   = PINNProcesses['M'  ].numpy()[:, 0][PINNmask] / repeats
                    PINNMx_general  = PINNProcesses['dM' ].numpy()[:, 0][PINNmask] / repeats
                    PINNMxx_general = PINNProcesses['dM2'].numpy()[:, 0][PINNmask] / repeats
                else:
                    PINNt_general   += PINNProcesses['t'  ].numpy()[:, 0][PINNmask] / repeats
                    PINNx_general   += PINNProcesses['x'  ].numpy()[:, 0][PINNmask] / repeats
                    PINNpi_general  += PINNProcesses['pi' ].numpy()[:, 0][PINNmask] / repeats
                    PINNM_general   += PINNProcesses['M'  ].numpy()[:, 0][PINNmask] / repeats
                    PINNMx_general  += PINNProcesses['dM' ].numpy()[:, 0][PINNmask] / repeats
                    PINNMxx_general+= PINNProcesses['dM2'].numpy()[:, 0][PINNmask]  / repeats
                
                
        if primal_general_nonconcave:
            repeats = num_repeats['primal_general_nonconcave']
            for n in range(repeats):
                log(f'Run {n}')
                tf.keras.backend.clear_session()
                tf.keras.backend.set_floatx('float64')
                PINNdata, PINNProcesses, PINNProcessesB = PINN_generalSolver(config_nonconcave).train()
                
                t0   = PINNProcesses['t'  ].numpy()[:, 0]
                x0   = PINNProcesses['x'  ].numpy()[:, 0]
                
                PINNmask = (x0 >= config.x_range[0]) & (x0 <= config.x_range[1]) #& np.isclose(t0, method.test_time)
                    
                if n == 0:
                    PINNt_general_nonconcave   = PINNProcesses['t'  ].numpy()[:, 0][PINNmask] / repeats
                    PINNx_general_nonconcave   = PINNProcesses['x'  ].numpy()[:, 0][PINNmask] / repeats
                    PINNpi_general_nonconcave  = PINNProcesses['pi' ].numpy()[:, 0][PINNmask] / repeats
                    PINNM_general_nonconcave   = PINNProcesses['M'  ].numpy()[:, 0][PINNmask]  / repeats
                    PINNMx_general_nonconcave  = PINNProcesses['dM' ].numpy()[:, 0][PINNmask]  / repeats
                    PINNMxx_general_nonconcave = PINNProcesses['dM2'].numpy()[:, 0][PINNmask]  / repeats
                else:
                    PINNt_general_nonconcave   += PINNProcesses['t'  ].numpy()[:, 0][PINNmask] / repeats
                    PINNx_general_nonconcave   += PINNProcesses['x'  ].numpy()[:, 0][PINNmask] / repeats
                    PINNpi_general_nonconcave  += PINNProcesses['pi' ].numpy()[:, 0][PINNmask] / repeats
                    PINNM_general_nonconcave   += PINNProcesses['M'  ].numpy()[:, 0][PINNmask]  / repeats
                    PINNMx_general_nonconcave  += PINNProcesses['dM' ].numpy()[:, 0][PINNmask]  / repeats
                    PINNMxx_general_nonconcave += PINNProcesses['dM2'].numpy()[:, 0][PINNmask]  / repeats
        
        
        if dual:
            repeats = num_repeats['dual']
            for n in range(repeats):
                log(f'Run {n}')
                loss = 1.0
                while loss > 1e-1:
                    tf.keras.backend.clear_session()
                    tf.keras.backend.set_floatx('float64')
                    PINNDualdata, PINNDualProcesses, PINNDualProcessesB = PINNDualSolver(config).train()
                    loss = PINNDualdata['M losses'][-1]
                                
                if n == 0:
                    Dualt   = PINNDualProcesses['t'  ].numpy()[:, 0] / repeats
                    Dualx   = PINNDualProcesses['x'  ].numpy()[:, 0] / repeats
                    Dualpi  = PINNDualProcesses['pi' ].numpy()[:, 0] / repeats
                    DualM   = PINNDualProcesses['M'  ].numpy()[:, 0] * np.exp(config.discount * (config.T - Dualt)) / repeats
                    DualMx  = PINNDualProcesses['dM' ].numpy()[:, 0] * np.exp(config.discount * (config.T - Dualt)) / repeats
                    DualMxx = PINNDualProcesses['dM2'].numpy()[:, 0] * np.exp(config.discount * (config.T - Dualt)) / repeats
                else:
                    Dualt   += PINNDualProcesses['t'  ].numpy()[:, 0] / repeats
                    Dualx   += PINNDualProcesses['x'  ].numpy()[:, 0] / repeats
                    Dualpi  += PINNDualProcesses['pi' ].numpy()[:, 0] / repeats
                    DualM   += PINNDualProcesses['M'  ].numpy()[:, 0] * np.exp(config.discount * (config.T - Dualt)) / repeats
                    DualMx  += PINNDualProcesses['dM' ].numpy()[:, 0] * np.exp(config.discount * (config.T - Dualt)) / repeats
                    DualMxx += PINNDualProcesses['dM2'].numpy()[:, 0] * np.exp(config.discount * (config.T - Dualt)) / repeats
                    
                    
            Dualmask = (Dualx >= config.x_range[0]) & (Dualx <= config.x_range[1]) #& np.isclose(Dualt, method.test_time)
            Dualt   = Dualt[Dualmask]
            Dualx   = Dualx[Dualmask]
            Dualpi  = Dualpi[Dualmask]
            DualM   = DualM[Dualmask]
            DualMx  = DualMx[Dualmask] 
            DualMxx = DualMxx[Dualmask] 
        
        if dual_general:
            repeats = num_repeats['dual_general']
            for n in range(repeats):
                log(f'Run {n}')
                loss = 1.0
                while loss > 1e-1:
                    tf.keras.backend.clear_session()
                    tf.keras.backend.set_floatx('float64')
                    PINNDualdata, PINNDualProcesses, PINNDualProcessesB = PINNDual_generalSolver(config).train()
                    loss = PINNDualdata['M losses'][-1]
                                
                if n == 0:
                    Dualt_general   = PINNDualProcesses['t'  ].numpy()[:, 0] / repeats
                    Dualx_general   = PINNDualProcesses['x'  ].numpy()[:, 0] / repeats
                    Dualpi_general  = PINNDualProcesses['pi' ].numpy()[:, 0] / repeats
                    DualM_general   = PINNDualProcesses['M'  ].numpy()[:, 0]  / repeats
                    DualMx_general  = PINNDualProcesses['dM' ].numpy()[:, 0]  / repeats
                    DualMxx_general = PINNDualProcesses['dM2'].numpy()[:, 0]  / repeats
                else:
                    Dualt_general   += PINNDualProcesses['t'  ].numpy()[:, 0] / repeats
                    Dualx_general   += PINNDualProcesses['x'  ].numpy()[:, 0] / repeats
                    Dualpi_general  += PINNDualProcesses['pi' ].numpy()[:, 0] / repeats
                    DualM_general   += PINNDualProcesses['M'  ].numpy()[:, 0]  / repeats
                    DualMx_general  += PINNDualProcesses['dM' ].numpy()[:, 0]  / repeats
                    DualMxx_general += PINNDualProcesses['dM2'].numpy()[:, 0]  / repeats
                    
                    
            Dualmask = (Dualx_general >= config.x_range[0]) & (Dualx_general <= config.x_range[1]) #& np.isclose(Dualt, method.test_time)
            Dualt_general   = Dualt_general[Dualmask]
            Dualx_general   = Dualx_general[Dualmask]
            Dualpi_general  = Dualpi_general[Dualmask]
            DualM_general   = DualM_general[Dualmask]
            DualMx_general  = DualMx_general[Dualmask] 
            DualMxx_general = DualMxx_general[Dualmask] 
        
        if smp:
            repeats = num_repeats['smp']
            for n in range(repeats):
                log(f'Run {n + 1}')
                tf.keras.backend.clear_session()
                tf.keras.backend.set_floatx('float64')
                SMPdata, SMPProcesses = SMPSolver(config).train()        
            
                t0   = SMPProcesses['t'  ].numpy()[:, 0]
                x0   = SMPProcesses['x'  ].numpy()[:, 0]
                pi0  = SMPProcesses['pi' ].numpy()[:, 0]
                
                monteX, _, monteM, _, _ = evaluate(pi0, t0, x0, config, method)
                SMPmask = (x0 >= config.x_range[0]) & (x0 <= config.x_range[1]) & np.isclose(t0, method.test_time)
                
                if n == 0:
                    SMPt   = SMPProcesses['t'  ].numpy()[:, 0][SMPmask] / repeats
                    SMPx   = SMPProcesses['x'  ].numpy()[:, 0][SMPmask] / repeats
                    SMPpi  = SMPProcesses['pi' ].numpy()[:, 0][SMPmask] / repeats
                    SMPMx  = SMPProcesses['dM' ].numpy()[:, 0][SMPmask] / repeats
                    SMPM   = monteM
                else:
                    SMPt   += SMPProcesses['t'  ].numpy()[:, 0][SMPmask] / repeats
                    SMPx   += SMPProcesses['x'  ].numpy()[:, 0][SMPmask] / repeats
                    SMPpi  += SMPProcesses['pi' ].numpy()[:, 0][SMPmask] / repeats
                    SMPMx  += SMPProcesses['dM' ].numpy()[:, 0][SMPmask] / repeats
                    SMPM = np.maximum(SMPM, monteM)
                    
                    
        if smp_nonconcave:
            repeats = num_repeats['smp_nonconcave']
            for n in range(repeats):
                log(f'Run {n}')
                tf.keras.backend.clear_session()
                tf.keras.backend.set_floatx('float64')
                SMPdata, SMPProcesses = SMPSolver(config_nonconcave).train()        
            
                t0   = SMPProcesses['t'  ].numpy()[:, 0]
                x0   = SMPProcesses['x'  ].numpy()[:, 0]
                pi0  = SMPProcesses['pi' ].numpy()[:, 0]
                
                monteX_nonconcave, _, monteM_nonconcave, _, _ = evaluate(pi0, t0, x0, config_nonconcave, method)
                SMPmask_nonconcave = (x0 >= config.x_range[0]) & (x0 <= config.x_range[1]) & np.isclose(t0, method.test_time)
                
                if n == 0:
                    SMPt_nonconcave   = SMPProcesses['t'  ].numpy()[:, 0][SMPmask_nonconcave] / repeats
                    SMPx_nonconcave   = SMPProcesses['x'  ].numpy()[:, 0][SMPmask_nonconcave] / repeats
                    SMPpi_nonconcave  = SMPProcesses['pi' ].numpy()[:, 0][SMPmask_nonconcave] / repeats
                    SMPMx_nonconcave  = SMPProcesses['dM' ].numpy()[:, 0][SMPmask_nonconcave] / repeats
                    SMPM_nonconcave   = monteM_nonconcave
                else:
                    SMPt_nonconcave   += SMPProcesses['t'  ].numpy()[:, 0][SMPmask_nonconcave] / repeats
                    SMPx_nonconcave   += SMPProcesses['x'  ].numpy()[:, 0][SMPmask_nonconcave] / repeats
                    SMPpi_nonconcave  += SMPProcesses['pi' ].numpy()[:, 0][SMPmask_nonconcave] / repeats
                    SMPMx_nonconcave  += SMPProcesses['dM' ].numpy()[:, 0][SMPmask_nonconcave] / repeats  
                    SMPM_nonconcave = np.maximum(SMPM_nonconcave, monteM_nonconcave)
        #solution
    
        xaxis = np.linspace(config.x_range[0], config.x_range[1], method.sample_size)
        yaxis = np.linspace(0.01, 2.0, method.sample_size)
        mid = (config.x_range[1] + config.x_range[0]) / 2
        length = config.x_range[1] - config.x_range[0]
        if config.lam * config.lower == 0:
            yaxis0 = yaxis[np.abs(-solution.d_v(method.test_time, yaxis) - mid) < length / 2] * np.exp(config.discount * (config.T - method.test_time) )
            states0 = -solution.d_v(method.test_time, yaxis0)
            controls0 = states0 * solution.pi(method.test_time, yaxis0)
            values0 = np.array([solution.value(method.test_time, x) for x in xaxis]) * np.exp(config.discount * (config.T - method.test_time) )
            derivs20 = solution.d2_v(method.test_time, yaxis0) * np.exp(config.discount * (config.T - method.test_time) )
            xaxis0 = xaxis
        else:
            values0 = solution.values 
            states0 = solution.states
            xaxis0 = solution.states
            yaxis0 = solution.yaxis 
    
        fig, axes = plt.subplots(2, 2, figsize = (12, 12))
        if primal:
            axes[0, 0].plot(PINNx, PINNpi, label = 'Primal PINN, $\\bar{U}$')
        if primal_nonconcave:
            axes[0, 0].plot(PINNx_nonconcave, PINNpi_nonconcave, label = 'Primal PINN, $U$')
        if primal_general:
            axes[0, 0].plot(PINNx_general, PINNpi_general, label = 'Primal PINN, $\\bar{U}$ (gen)')
        if primal_general_nonconcave:
            axes[0, 0].plot(PINNx_general_nonconcave, PINNpi_general_nonconcave, label = 'Primal PINN, $U$ (gen)')
        if dual:
            axes[0, 0].plot(Dualx, Dualpi, label = 'Dual PINN')
        if dual_general:
            axes[0, 0].plot(Dualx_general, Dualpi_general, label = 'Dual PINN (gen)')
        # axes[0, 0].plot(BSDEx, BSDEpi, label = 'BSDE')
        if smp:
            axes[0, 0].plot(SMPx, SMPpi, label = 'SMP')
        if smp_nonconcave:
            axes[0, 0].plot(SMPx_nonconcave, SMPpi_nonconcave, label = 'SMP, $U$')
        
        if config.lam * config.lower == 0 and soln:
            axes[0, 0].plot(states0, controls0, label = 'Solution', c = 'r', linestyle = 'dashed')
    
        axes[0, 0].set_xlabel('$x$')
        axes[0, 0].set_ylabel('$\pi(0, x, 1)$')
    
        axes[0, 0].set_title('control vs x')
        axes[0, 0].grid(0.5)
        axes[0, 0].legend()
        
        axes[0, 1].set_title('$v$ vs x')    
        axes[0, 1].grid(0.5)
        if primal:
            axes[0, 1].plot(PINNx, PINNM, label = 'Primal PINN, $\\bar{U}$')
        if primal_nonconcave:
            axes[0, 1].plot(PINNx_nonconcave, PINNM_nonconcave, label = 'Primal PINN, $U$')
        if primal_general:
            axes[0, 1].plot(PINNx_general, PINNM_general, label = 'Primal PINN, $\\bar{U}$ (gen)')
        if primal_general_nonconcave:
            axes[0, 1].plot(PINNx_general_nonconcave, PINNM_general_nonconcave, label = 'Primal PINN, $U$ (gen)')
        if dual:
            axes[0, 1].plot(Dualx, DualM, label = 'Dual PINN')
        if dual_general:
            axes[0, 1].plot(Dualx_general, DualM_general, label = 'Dual PINN (gen)')
        if smp:
            axes[0, 1].plot(monteX, SMPM, label = 'SMP')
        if smp_nonconcave:
            axes[0, 1].plot(monteX_nonconcave, SMPM_nonconcave, label = 'SMP, $U$')
            
        if soln:
            axes[0, 1].plot(xaxis0, values0, label = 'Solution', c = 'r', linestyle = 'dashed')
        axes[0, 1].set_xlabel('$x$')
        axes[0, 1].set_ylabel('$v(0, x, 1)$')
        axes[0, 1].legend()
    
        axes[1, 0].set_title('$\partial_xv$ vs x')    
        axes[1, 0].grid(0.5)
        if primal:
            axes[1, 0].plot(PINNx, PINNMx, label = 'Primal PINN, $\\bar{U}$')
        if primal_nonconcave:
            axes[1, 0].plot(PINNx_nonconcave, PINNMx_nonconcave, label = 'Primal PINN, $U$')
        if primal_general:
            axes[1, 0].plot(PINNx_general, PINNMx_general, label = 'Primal PINN, $\\bar{U}$ (gen)')
        if primal_general_nonconcave:
            axes[1, 0].plot(PINNx_general_nonconcave, PINNMx_general_nonconcave, label = 'Primal PINN, $U$ (gen)')
        if dual:
            axes[1, 0].plot(Dualx, DualMx, label = 'Dual PINN')
        if dual_general:
            axes[1, 0].plot(Dualx_general, DualMx_general, label = 'Dual PINN (gen)')
        if smp:
            axes[1, 0].plot(SMPx, SMPMx, label = 'SMP')
        if smp_nonconcave:
            axes[1, 0].plot(SMPx_nonconcave, SMPMx_nonconcave, label = 'SMP, $U$')
        if soln:
            axes[1, 0].plot(states0, yaxis0, label = 'Solution', c = 'r', linestyle = 'dashed')
        axes[1, 0].set_xlabel('$x$')
        axes[1, 0].set_ylabel('$\partial_xv(0, x, 1)$')
        axes[1, 0].legend()
        
        if primal:
            PINNmask2 = (PINNt < config.T - 1e-6) & (PINNMxx > threshold)
        if primal_nonconcave:
            PINNmask2_nonconcave = (PINNt_nonconcave < config.T - 1e-6) & (PINNMxx_nonconcave > threshold)
        if primal_general:
            PINNmask2_general = (PINNt_general < config.T - 1e-6) & (PINNMxx_general > threshold)
        if primal_general_nonconcave:
            PINNmask2_general_nonconcave = (PINNt_general_nonconcave < config.T - 1e-6) & (PINNMxx_general_nonconcave > threshold)
        if dual:
            Dualmask2 = (Dualt < config.T - 1e-6) & (DualMxx > threshold) & (DualMxx < 1.0)
        if dual_general:
            Dualmask2_general = (Dualt_general < config.T - 1e-6) & (DualMxx_general > threshold) & (DualMxx_general < 1.0)
    
        axes[1, 1].set_title('$\partial_{xx}v$ vs x')          
        axes[1, 1].grid(0.5)
        if primal:
            axes[1, 1].plot(PINNx[PINNmask2], PINNMxx[PINNmask2], label = 'Primal PINN, $\\bar{U}$')
        if primal_nonconcave:
            axes[1, 1].plot(PINNx_nonconcave[PINNmask2_nonconcave], PINNMxx_nonconcave[PINNmask2_nonconcave], label = 'Primal PINN, $U$')
        if primal_general:
            axes[1, 1].plot(PINNx_general[PINNmask2_general], PINNMxx_general[PINNmask2_general], label = 'Primal PINN, $\\bar{U}$ (gen)')
        if primal_general_nonconcave:
            axes[1, 1].plot(PINNx_general_nonconcave[PINNmask2_general_nonconcave], PINNMxx_general_nonconcave[PINNmask2_general_nonconcave], label = 'Primal PINN, $U$ (gen)')
        if dual:
            axes[1, 1].plot(Dualx[Dualmask2], DualMxx[Dualmask2], label = 'Dual PINN')
        if dual_general:
            axes[1, 1].plot(Dualx_general[Dualmask2_general], DualMxx_general[Dualmask2_general], label = 'Dual PINN (gen)')
        if config.lam * config.lower == 0 and soln:
            axes[1, 1].plot(states0[derivs20 > 0.01], -1 / derivs20[derivs20 > 0.01], label = 'Reference', c = 'r', linestyle = 'dashed')
        axes[1, 1].set_xlabel('$x$')
        axes[1, 1].set_ylabel('$\partial_{xx}v(0, x, 1)$')
        axes[1, 1].legend()
            
                
        if soln:
            solF = lambda x: np.interp(x, xaxis0, values0)
              
        
            
            plt.figure()
            if primal:
                plt.plot(PINNx, PINNM - solF(PINNx), label = 'Primal PINN, $\\bar{U}$')
            if primal_nonconcave:
                plt.plot(PINNx_nonconcave, PINNM_nonconcave - solF(PINNx_nonconcave), label = 'Primal PINN, $U$')
            if primal_general:
                plt.plot(PINNx_general, PINNM_general - solF(PINNx_general), label = 'Primal PINN, $\\bar{U}$ (gen)')
            if primal_general_nonconcave:
                plt.plot(PINNx_general_nonconcave, PINNM_general_nonconcave - solF(PINNx_general_nonconcave), label = 'Primal PINN, $U$ (gen)')
            if dual:
                plt.plot(Dualx, DualM - solF(Dualx), label = 'Dual PINN')
            if dual_general:
                plt.plot(Dualx_general, DualM_general - solF(Dualx_general), label = 'Dual PINN (gen)')
            if smp:
                plt.plot(monteX, monteM - solF(monteX), label = 'SMP')
            if smp_nonconcave:
                plt.plot(monteX_nonconcave, monteM_nonconcave - solF(monteX_nonconcave), label = 'SMP, $U$')
            plt.plot(xaxis0, np.array(values0) - solF(xaxis0), label = 'Solution', c = 'r', linestyle = 'dotted')
            plt.xlabel('$x$')
            plt.ylabel('Absolute Error')
            plt.grid(0.5)
            plt.legend()
            plt.show()    
                        

    
    
if __name__ == '__main__':
    main()
    
    