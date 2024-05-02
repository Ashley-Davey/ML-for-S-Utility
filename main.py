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


        
    r_range_final = (1.0, 1.0)
    x_range_final = (0.05, 5.0)
    y_range_final = (0.15, 2.0)
    
    for rho, scaling in [ [1.0, True]]:
        r_test = r_range_final[0] if r_range_final[0] == r_range_final[1] else 1.0

        print(rho, scaling)
        config = Problem(non_concave = False, r_range_final = r_range_final, x_range_final = x_range_final, y_range_final = y_range_final, rho = rho, scaling = scaling)
        config_nonconcave = Problem(non_concave = True, r_range_final = r_range_final, x_range_final = x_range_final, y_range_final = y_range_final, rho = rho, scaling = scaling)
    
    
        primal = True#config.scaling
        primal_nonconcave = True#config.scaling
        primal_general = True#not config.scaling
        primal_general_nonconcave = True#not config.scaling
        dual = True#config.scaling
        dual_general = True#not config.scaling
        smp = True
        # smp_nonconcave = False
        soln = (config.rho == 1.0) and config.scaling
                
        
        method = Method(config)
        solution = Solution(config)
        
        #print solutions
        
        yaxis = np.linspace(config.y_range_sol[0]*0.9, config.y_range_sol[1], method.sample_size)
        mid = (config.x_range_final[1]*1.1 + config.x_range_final[0]) / 2
        length = config.x_range_final[1]*1.1  - config.x_range_final[0]
        
        yaxis0 = yaxis[np.abs(-solution.d_v(method.test_time, yaxis) - mid) < length / 2] * np.exp(config.discount * (config.T - method.test_time) )
        xaxis0 = -solution.d_v(method.test_time, yaxis0) 
        values0 = np.array([solution.value(method.test_time, x) for x in xaxis0]) * np.exp(config.discount * (config.T - method.test_time) ) 
        
        eval_mask = (np.isclose(xaxis0,0.5,atol=1e-2) | np.isclose(xaxis0,1.0,atol=1e-2)) | np.isclose(xaxis0,5.0,atol=1e-2)
        
        X = xaxis0[eval_mask]
        M = values0[eval_mask]
        
        # print(X, M)
        
        for x, m in zip(X, M):
            print(x, 1.0, m)
        
        x = 1.0
        raxis = np.array([0.5, 1.0, 5.0])
        values0 = np.array([solution.value(method.test_time, x / r) for r in raxis]) * np.exp(config.discount * (config.T - method.test_time) ) * np.power(raxis, config.p)
        R = raxis
        M = values0
        
        # print(R, M)
    
        for r, m in zip(R, M):
            print(1.0, r, m)
            
            
        if primal:
            repeats = num_repeats['primal']
            for n in range(repeats):
                log(f'Run {n + 1}')
                tf.keras.backend.clear_session()
                tf.keras.backend.set_floatx('float64')
                data, PINNProcesses, PINNProcessesB = PINNSolver(config).train()
                if len(data['M losses']) > 0:
                    fig, axes = plt.subplots(1)
                    axes.plot(data['M losses'])
                    axes.set_title('primal' + ' Loss')
                    axes.set_yscale('log')
                    axes.set_xlabel('Iteration Step')
                    axes.set_ylabel('Loss')
                    axes.grid(0.5)    
                    plt.show()
                
                    
                if n == 0:
                    PINNt   = PINNProcesses['t'  ].numpy()[:, 0] / repeats
                    PINNx   = PINNProcesses['x'  ].numpy()[:, 0] / repeats
                    PINNpi  = PINNProcesses['pi' ].numpy()[:, 0] / repeats
                    PINNM   = PINNProcesses['M'  ].numpy()[:, 0] * np.exp(config.discount * (config.T - PINNt)) / repeats
                    PINNMx  = PINNProcesses['dM' ].numpy()[:, 0] * np.exp(config.discount * (config.T - PINNt)) / repeats
                    PINNMxx = PINNProcesses['dM2'].numpy()[:, 0] * np.exp(config.discount * (config.T - PINNt)) / repeats
                else:
                    PINNt   += PINNProcesses['t'  ].numpy()[:, 0] / repeats
                    PINNx   += PINNProcesses['x'  ].numpy()[:, 0] / repeats
                    PINNpi  += PINNProcesses['pi' ].numpy()[:, 0] / repeats
                    PINNM   += PINNProcesses['M'  ].numpy()[:, 0] * np.exp(config.discount * (config.T - PINNt)) / repeats
                    PINNMx  += PINNProcesses['dM' ].numpy()[:, 0] * np.exp(config.discount * (config.T - PINNt)) / repeats
                    PINNMxx += PINNProcesses['dM2'].numpy()[:, 0] * np.exp(config.discount * (config.T - PINNt)) / repeats
                    
                    
            targets = [0.2, 0.5, 1.0, 2.0, 5.0]
            X = PINNx
            M = PINNM
            
            for x in targets:
                score = np.power(X - x, 2)
                i = np.argmin(score)
                if score[i] < 1e-2:

                    print(X[i], 1.0, M[i])
                
            PINNmask = (PINNx >= config.x_range_final[0]) & (PINNx <= config.x_range_final[1]) 
            
            i = np.argsort(PINNx[PINNmask])
            
            PINNt   = PINNt  [PINNmask][i]
            PINNx   = PINNx  [PINNmask][i]
            PINNpi  = PINNpi [PINNmask][i]
            PINNM   = PINNM  [PINNmask][i]
            PINNMx  = PINNMx [PINNmask][i]
            PINNMxx = PINNMxx[PINNmask][i]
                    
    
                
                
                
                
        if primal_nonconcave:
            repeats = num_repeats['primal_nonconcave']
            for n in range(repeats):
                log(f'Run {n+1}')
                tf.keras.backend.clear_session()
                tf.keras.backend.set_floatx('float64')
                data, PINNProcesses, PINNProcessesB = PINNSolver(config_nonconcave).train()
                if len(data['M losses']) > 0:
                    fig, axes = plt.subplots(1)
                    axes.plot(data['M losses'])
                    axes.set_title('primal NC' + ' Loss')
                    axes.set_yscale('log')
                    axes.set_xlabel('Iteration Step')
                    axes.set_ylabel('Loss')
                    axes.grid(0.5)    
                    plt.show()
                    
                if n == 0:
                    PINNt_nonconcave   = PINNProcesses['t'  ].numpy()[:, 0] / repeats
                    PINNx_nonconcave   = PINNProcesses['x'  ].numpy()[:, 0] / repeats
                    PINNpi_nonconcave  = PINNProcesses['pi' ].numpy()[:, 0] / repeats
                    PINNM_nonconcave   = PINNProcesses['M'  ].numpy()[:, 0] * np.exp(config.discount * (config.T - PINNt_nonconcave)) / repeats
                    PINNMx_nonconcave  = PINNProcesses['dM' ].numpy()[:, 0] * np.exp(config.discount * (config.T - PINNt_nonconcave)) / repeats
                    PINNMxx_nonconcave = PINNProcesses['dM2'].numpy()[:, 0] * np.exp(config.discount * (config.T - PINNt_nonconcave)) / repeats
                else:
                    PINNt_nonconcave   += PINNProcesses['t'  ].numpy()[:, 0] / repeats
                    PINNx_nonconcave   += PINNProcesses['x'  ].numpy()[:, 0] / repeats
                    PINNpi_nonconcave  += PINNProcesses['pi' ].numpy()[:, 0] / repeats
                    PINNM_nonconcave   += PINNProcesses['M'  ].numpy()[:, 0] * np.exp(config.discount * (config.T - PINNt_nonconcave)) / repeats
                    PINNMx_nonconcave  += PINNProcesses['dM' ].numpy()[:, 0] * np.exp(config.discount * (config.T - PINNt_nonconcave)) / repeats
                    PINNMxx_nonconcave += PINNProcesses['dM2'].numpy()[:, 0] * np.exp(config.discount * (config.T - PINNt_nonconcave)) / repeats
        
        
            targets = [0.2, 0.5, 1.0, 2.0, 5.0]
            X = PINNx_nonconcave
            M = PINNM_nonconcave
            
            for x in targets:
                score = np.power(X - x, 2)
                i = np.argmin(score)
                if score[i] < 1e-2:

                    print(X[i], 1.0, M[i])
            
            PINNmask = (PINNx_nonconcave >= config.x_range_final[0]) & (PINNx_nonconcave <= config.x_range_final[1]) #& np.isclose(t0, method.test_time)
            i = np.argsort(PINNx_nonconcave[PINNmask])

            PINNt_nonconcave   = PINNt_nonconcave  [PINNmask][i]
            PINNx_nonconcave   = PINNx_nonconcave  [PINNmask][i]
            PINNpi_nonconcave  = PINNpi_nonconcave [PINNmask][i]
            PINNM_nonconcave   = PINNM_nonconcave  [PINNmask][i]
            PINNMx_nonconcave  = PINNMx_nonconcave [PINNmask][i]
            PINNMxx_nonconcave = PINNMxx_nonconcave[PINNmask][i]
        
        if primal_general:
            repeats = num_repeats['primal_general']
            for n in range(repeats):
                log(f'Run {n + 1}')
                tf.keras.backend.clear_session()
                tf.keras.backend.set_floatx('float64')
                loss = 1.0
                while loss > 1e-2:
                    tf.keras.backend.clear_session()
                    tf.keras.backend.set_floatx('float64')
                    data, PINNProcesses, PINNProcessesB = PINN_generalSolver(config).train()
                    loss = data['M losses'][-1]  if Method(config, 'pinn_general').iteration_steps > 10000 else 0 

                if len(data['M losses']) > 0:
                    fig, axes = plt.subplots(1)
                    axes.plot(data['M losses'])
                    axes.set_title('primal gen' + ' Loss')
                    axes.set_yscale('log')
                    axes.set_xlabel('Iteration Step')
                    axes.set_ylabel('Loss')
                    axes.grid(0.5)    
                    plt.show()
                    
                if n == 0:
                    PINNt_general   = PINNProcesses['t'  ].numpy()[:, 0] / repeats
                    PINNx_general   = PINNProcesses['x'  ].numpy()[:, 0] / repeats
                    PINNr_general   = PINNProcesses['r'  ].numpy()[:, 0] / repeats
                    PINNpi_general  = PINNProcesses['pi' ].numpy()[:, 0] / repeats
                    PINNM_general   = PINNProcesses['M'  ].numpy()[:, 0] / repeats
                    PINNMx_general  = PINNProcesses['dM' ].numpy()[:, 0] / repeats
                    PINNMxx_general = PINNProcesses['dM2'].numpy()[:, 0] / repeats
                else:
                    PINNt_general   += PINNProcesses['t'  ].numpy()[:, 0] / repeats
                    PINNx_general   += PINNProcesses['x'  ].numpy()[:, 0] / repeats
                    PINNr_general   += PINNProcesses['r'  ].numpy()[:, 0] / repeats
                    PINNpi_general  += PINNProcesses['pi' ].numpy()[:, 0] / repeats
                    PINNM_general   += PINNProcesses['M'  ].numpy()[:, 0] / repeats
                    PINNMx_general  += PINNProcesses['dM' ].numpy()[:, 0] / repeats
                    PINNMxx_general += PINNProcesses['dM2'].numpy()[:, 0] / repeats
                    
                    
                    
            targets = [[0.5, 1.0], [1.0, 1.0], [5.0, 1.0], [1.0, 0.5], [1.0, 5.0]]
            X = PINNx_general
            R = PINNr_general
            M = PINNM_general
            
            for x, r in targets:
                score = np.power(X - x, 2) + np.power(R - r, 2)
                i = np.argmin(score)
                if score[i] < 1e-2:
                    print(X[i], R[i], M[i])
                    
                
            PINNmask_r = np.isclose(PINNx_general, 1.0)
            i = np.argsort(PINNr_general[PINNmask_r])
            PINNr_general_r   = PINNr_general[PINNmask_r][i]
            PINNpi_general_r  = PINNpi_general[PINNmask_r][i]
            PINNM_general_r   = PINNM_general[PINNmask_r][i]
                
            PINNmask = np.isclose(PINNr_general, r_test)
            
            i = np.argsort(PINNx_general[PINNmask])
            
            PINNt_general  = PINNt_general[PINNmask][i]
            PINNx_general  = PINNx_general[PINNmask][i]
            PINNr_general  = PINNr_general[PINNmask][i]
            PINNpi_general = PINNpi_general[PINNmask][i]
            PINNM_general   = PINNM_general[PINNmask][i]
            PINNMx_general  = PINNMx_general[PINNmask][i]
            PINNMxx_general = PINNMxx_general[PINNmask][i]
                                    
        if primal_general_nonconcave:
            repeats = num_repeats['primal_general_nonconcave']
            for n in range(repeats):
                log(f'Run {n + 1}')
                tf.keras.backend.clear_session()
                tf.keras.backend.set_floatx('float64')
                data, PINNProcesses, PINNProcessesB = PINN_generalSolver(config_nonconcave).train()
                    
                if len(data['M losses']) > 0:
                    fig, axes = plt.subplots(1)
                    axes.plot(data['M losses'])
                    axes.set_title('primal gen NC' + ' Loss')
                    axes.set_yscale('log')
                    axes.set_xlabel('Iteration Step')
                    axes.set_ylabel('Loss')
                    axes.grid(0.5)   
                    plt.show()
                    
                if n == 0:
                    PINNt_general_nonconcave   = PINNProcesses['t'  ].numpy()[:, 0] / repeats
                    PINNx_general_nonconcave   = PINNProcesses['x'  ].numpy()[:, 0] / repeats
                    PINNr_general_nonconcave   = PINNProcesses['r'  ].numpy()[:, 0] / repeats
                    PINNpi_general_nonconcave  = PINNProcesses['pi' ].numpy()[:, 0] / repeats
                    PINNM_general_nonconcave   = PINNProcesses['M'  ].numpy()[:, 0] / repeats
                    PINNMx_general_nonconcave  = PINNProcesses['dM' ].numpy()[:, 0] / repeats
                    PINNMxx_general_nonconcave = PINNProcesses['dM2'].numpy()[:, 0] / repeats
                else:
                    PINNt_general_nonconcave   += PINNProcesses['t'  ].numpy()[:, 0] / repeats
                    PINNx_general_nonconcave   += PINNProcesses['x'  ].numpy()[:, 0] / repeats
                    PINNr_general_nonconcave   += PINNProcesses['r'  ].numpy()[:, 0] / repeats
                    PINNpi_general_nonconcave  += PINNProcesses['pi' ].numpy()[:, 0] / repeats
                    PINNM_general_nonconcave   += PINNProcesses['M'  ].numpy()[:, 0] / repeats
                    PINNMx_general_nonconcave  += PINNProcesses['dM' ].numpy()[:, 0] / repeats
                    PINNMxx_general_nonconcave += PINNProcesses['dM2'].numpy()[:, 0] / repeats
                    
            targets = [[0.5, 1.0], [1.0, 1.0], [5.0, 1.0], [1.0, 0.5], [1.0, 5.0]]
            X = PINNx_general_nonconcave
            R = PINNr_general_nonconcave
            M = PINNM_general_nonconcave
            
            for x, r in targets:
                score = np.power(X - x, 2) + np.power(R - r, 2)
                i = np.argmin(score)
                if score[i] < 1e-2:
                    print(X[i], R[i], M[i])
    
            PINNmask_r = np.isclose(PINNx_general_nonconcave, 1.0)
            i = np.argsort(PINNr_general_nonconcave[PINNmask_r])    
            
            PINNr_general_nonconcave_r   = PINNr_general_nonconcave[PINNmask_r][i]
            PINNpi_general_nonconcave_r  = PINNpi_general_nonconcave[PINNmask_r][i]
            PINNM_general_nonconcave_r   = PINNM_general_nonconcave[PINNmask_r][i]
                
            PINNmask = np.isclose(PINNr_general_nonconcave, r_test)
            i = np.argsort(PINNx_general_nonconcave[PINNmask])
    
            PINNt_general_nonconcave  = PINNt_general_nonconcave[PINNmask][i]
            PINNx_general_nonconcave  = PINNx_general_nonconcave[PINNmask][i]
            PINNr_general_nonconcave  = PINNr_general_nonconcave[PINNmask][i]
            PINNpi_general_nonconcave = PINNpi_general_nonconcave[PINNmask][i]
            PINNM_general_nonconcave   = PINNM_general_nonconcave[PINNmask][i]
            PINNMx_general_nonconcave  = PINNMx_general_nonconcave[PINNmask][i]
            PINNMxx_general_nonconcave = PINNMxx_general_nonconcave[PINNmask][i]
                
                
        if dual:
            repeats = num_repeats['dual']
            for n in range(repeats):
                log(f'Run {n + 1}')
                loss = 1.0
                while loss > 1e-1:
                    tf.keras.backend.clear_session()
                    tf.keras.backend.set_floatx('float64')
                    data, PINNDualProcesses, PINNDualProcessesB = PINNDualSolver(config).train()
                    loss = data['M losses'][-1] if Method(config, 'dual').iteration_steps > 10000 else 0
                    
                if len(data['M losses']) > 0:
                    fig, axes = plt.subplots(1)
                    axes.plot(data['M losses'])
                    axes.set_title('dual' + ' Loss')
                    axes.set_yscale('log')
                    axes.set_xlabel('Iteration Step')
                    axes.set_ylabel('Loss')
                    axes.grid(0.5)    
                    plt.show()

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
                    
            targets = [0.2, 0.5, 1.0, 2.0, 5.0]
            X = Dualx
            M = DualM
            
            for x in targets:
                score = np.power(X - x, 2)
                i = np.argmin(score)
                if score[i] < 1e-2:

                    print(X[i], 1.0, M[i])
                    
                    
                    
            Dualmask = (Dualx >= config.x_range_final[0]) & (Dualx <= config.x_range_final[1]) #& np.isclose(Dualt, method.test_time)
            i = np.argsort(Dualx[Dualmask])
            
            
            
            Dualt   = Dualt[Dualmask][i]
            Dualx   = Dualx[Dualmask][i]
            Dualpi  = Dualpi[Dualmask][i]
            DualM   = DualM[Dualmask][i]
            DualMx  = DualMx[Dualmask][i] 
            DualMxx = DualMxx[Dualmask][i] 
            
    
    
        
        if dual_general:
            repeats = num_repeats['dual_general']
            for n in range(repeats):
                log(f'Run {n+1}')
                loss = 1.0
                while loss > 1e-1:
                    tf.keras.backend.clear_session()
                    tf.keras.backend.set_floatx('float64')
                    data, PINNDualProcesses, PINNDualProcessesB = PINNDual_generalSolver(config).train()
                    loss = data['M losses'][-1] if Method(config, 'dual_general').iteration_steps > 10000 else 0
                                
                if len(data['M losses']) > 0:
                    fig, axes = plt.subplots(1)
                    axes.plot(data['M losses'])
                    axes.set_title('dual gen' + ' Loss')
                    axes.set_yscale('log')
                    axes.set_xlabel('Iteration Step')
                    axes.set_ylabel('Loss')
                    axes.grid(0.5)    
                    plt.show()

                if n == 0:
                    Dualt_general   = PINNDualProcesses['t'  ].numpy()[:, 0] / repeats
                    Dualx_general   = PINNDualProcesses['x'  ].numpy()[:, 0] / repeats
                    Dualr_general   = PINNDualProcesses['r'  ].numpy()[:, 0] / repeats
                    Dualpi_general  = PINNDualProcesses['pi' ].numpy()[:, 0] / repeats
                    DualM_general   = PINNDualProcesses['M'  ].numpy()[:, 0]  / repeats
                    DualMx_general  = PINNDualProcesses['dM' ].numpy()[:, 0]  / repeats
                    DualMxx_general = PINNDualProcesses['dM2'].numpy()[:, 0]  / repeats
                else:
                    Dualt_general   += PINNDualProcesses['t'  ].numpy()[:, 0] / repeats
                    Dualx_general   += PINNDualProcesses['x'  ].numpy()[:, 0] / repeats
                    Dualr_general   += PINNDualProcesses['r'  ].numpy()[:, 0] / repeats
                    Dualpi_general  += PINNDualProcesses['pi' ].numpy()[:, 0] / repeats
                    DualM_general   += PINNDualProcesses['M'  ].numpy()[:, 0]  / repeats
                    DualMx_general  += PINNDualProcesses['dM' ].numpy()[:, 0]  / repeats
                    DualMxx_general += PINNDualProcesses['dM2'].numpy()[:, 0]  / repeats
                    
            
            targets = [[0.5, 1.0], [1.0, 1.0], [5.0, 1.0], [1.0, 0.5], [1.0, 5.0]]
            X = Dualx_general
            R = Dualr_general
            M = DualM_general
            
            for x, r in targets:
                score = np.power(X - x, 2) + np.power(R - r, 2)
                i = np.argmin(score)
                if score[i] < 1e-2:
                    print(X[i], R[i], M[i])
            
            Dualmask_r = np.isclose(Dualx_general, 1.0, atol = 1e-2)
            
            Dualr_general_r   = Dualr_general[Dualmask_r]
            i = np.argsort(Dualr_general_r)
            Dualr_general_r   = Dualr_general_r[i]
            Dualpi_general_r = Dualpi_general[Dualmask_r][i]
            DualM_general_r   = DualM_general[Dualmask_r][i]
            
            Dualmask = (Dualx_general >= config.x_range_final[0]) & (Dualx_general <= config.x_range_final[1]) & np.isclose(Dualr_general, r_test, atol = 1e-2)
            
            i = np.argsort(Dualx_general[Dualmask])
            
            
            Dualx_general   = Dualx_general[Dualmask][i]
            Dualt_general   = Dualt_general[Dualmask][i]
            Dualr_general   = Dualr_general[Dualmask][i]
            Dualpi_general  = Dualpi_general[Dualmask][i]
            DualM_general   = DualM_general[Dualmask][i]
            DualMx_general  = DualMx_general[Dualmask][i]
            DualMxx_general = DualMxx_general[Dualmask][i]
            
            
    
            
            
            
    
        
        if smp:
            repeats = num_repeats['smp']
            for n in range(repeats):
                log(f'Run {n + 1}')
                tf.keras.backend.clear_session()
                tf.keras.backend.set_floatx('float64')
                data, SMPProcesses = SMPSolver(config).train()        
                
                # if len(data['M losses']) > 0:
                #     fig, axes = plt.subplots(1)
                #     axes.plot(data['M losses'])
                #     axes.set_title('SMP' + ' Loss')
                #     axes.set_yscale('log')
                #     axes.set_xlabel('Iteration Step')
                #     axes.set_ylabel('Loss')
                #     axes.grid(0.5)    
                    
                #     if 'control losses' in data.keys():
                #         fig, axes = plt.subplots(1)
                #         axes.plot(data['control losses'])
                #         axes.set_title('Control Losses')
                #         axes.set_yscale('log')
                #         axes.set_xlabel('Iteration Step')
                #         axes.set_ylabel('Loss')
                #         axes.grid(0.5)
                #     plt.show()
                    
                    
                t0   = SMPProcesses['t'  ].numpy()[:, 0]
                x0   = SMPProcesses['x'  ].numpy()[:, 0]
                r0   = SMPProcesses['r'  ].numpy()[:, 0]
                pi0  = SMPProcesses['pi' ].numpy()[:, 0]
                
                monteX, _, monteM, _, monteR = evaluate(pi0, t0, x0, config, method)
                
                
                if n == 0:
                    SMPt   = SMPProcesses['t'  ].numpy()[:, 0] / repeats
                    SMPx   = SMPProcesses['x'  ].numpy()[:, 0] / repeats
                    SMPr   = SMPProcesses['r'  ].numpy()[:, 0] / repeats
                    SMPpi  = SMPProcesses['pi' ].numpy()[:, 0] / repeats
                    SMPMx  = SMPProcesses['dM' ].numpy()[:, 0] / repeats
                    SMPMxx  = SMPProcesses['dM2' ].numpy()[:, 0] / repeats
                    SMPM   = monteM / repeats
                else:
                    SMPt   += SMPProcesses['t'  ].numpy()[:, 0] / repeats
                    SMPx   += SMPProcesses['x'  ].numpy()[:, 0] / repeats
                    SMPr   += SMPProcesses['r'  ].numpy()[:, 0] / repeats
                    SMPpi  += SMPProcesses['pi' ].numpy()[:, 0] / repeats
                    SMPMx  += SMPProcesses['dM' ].numpy()[:, 0] / repeats
                    SMPMxx  += SMPProcesses['dM2' ].numpy()[:, 0] / repeats
                    SMPM += monteM / repeats
                
            
            targets = [[0.5, 1.0], [1.0, 1.0], [5.0, 1.0], [1.0, 0.5], [1.0, 5.0]]
            X = monteX
            R = monteR
            M = monteM
            
            for x, r in targets:
                score = np.power(X - x, 2) + np.power(R - r, 2)
                i = np.argmin(score)
                if score[i] < 1e-2:
                    print(X[i], R[i], M[i])

        
            
    
            SMPmask_r = np.isclose(SMPt, method.test_time) & np.isclose(SMPx, 1.0)
            Montemask_r = np.isclose(monteX, 1.0)
            
            i = np.argsort(SMPr[SMPmask_r])
            
            SMPr_r   = SMPr[SMPmask_r][i]
            SMPpi_r  = SMPpi[SMPmask_r][i]
            
            i = np.argsort(monteR[Montemask_r]   )
            
            SMPM_r   = monteM[Montemask_r][i]
            monteR_r = monteR[Montemask_r][i]                    
                            
            SMPmask = np.isclose(t0, method.test_time) & np.isclose(r0, r_test)
            Montemask = np.isclose(monteR, r_test)
            
            i = np.argsort(SMPx[SMPmask])
    
            SMPt   = SMPt[SMPmask][i]
            SMPx   = SMPx[SMPmask][i]
            SMPr   = SMPr[SMPmask][i]
            SMPpi  = SMPpi[SMPmask][i]
            SMPMx  = SMPMx[SMPmask][i]
            SMPMxx  = SMPMxx[SMPmask][i]
            
            i = np.argsort(monteX[Montemask])
            
            SMPM   = monteM[Montemask][i]
            monteX_x = monteX[Montemask][i]
                
                
                    
        # if smp_nonconcave:
        #     repeats = num_repeats['smp_nonconcave']
        #     for n in range(repeats):
        #         log(f'Run {n+1}')
        #         tf.keras.backend.clear_session()
        #         tf.keras.backend.set_floatx('float64')
        #         SMPdata, SMPProcesses = SMPSolver(config_nonconcave).train()        
            
        #         t0   = SMPProcesses['t'  ].numpy()[:, 0]
        #         x0   = SMPProcesses['x'  ].numpy()[:, 0]
        #         pi0  = SMPProcesses['pi' ].numpy()[:, 0]
        #         r0   = SMPProcesses['r' ].numpy()[:, 0]
                
        #         monteX_nonconcave, _, monteM_nonconcave, _, _ = evaluate(pi0, t0, x0, config_nonconcave, method)
        #         SMPmask_nonconcave = (x0 >= config.x_range_final[0]) & (x0 <= config.x_range_final[1]) & np.isclose(t0, method.test_time)
                
        #         if n == 0:
        #             SMPt_nonconcave   = SMPProcesses['t'  ].numpy()[:, 0][SMPmask_nonconcave] / repeats
        #             SMPx_nonconcave   = SMPProcesses['x'  ].numpy()[:, 0][SMPmask_nonconcave] / repeats
        #             SMPpi_nonconcave  = SMPProcesses['pi' ].numpy()[:, 0][SMPmask_nonconcave] / repeats
        #             SMPMx_nonconcave  = SMPProcesses['dM' ].numpy()[:, 0][SMPmask_nonconcave] / repeats
        #             SMPM_nonconcave   = monteM_nonconcave
        #         else:
        #             SMPt_nonconcave   += SMPProcesses['t'  ].numpy()[:, 0][SMPmask_nonconcave] / repeats
        #             SMPx_nonconcave   += SMPProcesses['x'  ].numpy()[:, 0][SMPmask_nonconcave] / repeats
        #             SMPpi_nonconcave  += SMPProcesses['pi' ].numpy()[:, 0][SMPmask_nonconcave] / repeats
        #             SMPMx_nonconcave  += SMPProcesses['dM' ].numpy()[:, 0][SMPmask_nonconcave] / repeats  
        #             SMPM_nonconcave = np.maximum(SMPM_nonconcave, monteM_nonconcave)
        #solution
        
        if ((r_range_final[0] <= 1.0 and r_range_final[1] >= 1.0) or r_range_final[0] == r_range_final[1]) and x_range_final[0] != x_range_final[1]:
            r = r_range_final[0] if r_range_final[0] == r_range_final[1] else 1.0
            
            yaxis = np.linspace(config.y_range_sol[0], config.y_range_sol[1], method.sample_size)
            mid = (config.x_range_final[1] / r + config.x_range_final[0] / r) / 2
            length = config.x_range_final[1] / r - config.x_range_final[0] / r
            
            ys = yaxis[np.abs(-solution.d_v(method.test_time, yaxis) - mid) < length / 2]
            yaxis0 =  ys* np.exp(config.discount * (config.T - method.test_time) ) * np.power(r_range_final[0], config.p - 1)
            states0 = -solution.d_v(method.test_time, ys) * r
            values0 = np.array([solution.value(method.test_time, x / r) for x in states0]) * np.exp(config.discount * (config.T - method.test_time) ) * np.power(r, config.p)
            derivs20 = solution.d2_v(method.test_time, ys) * np.exp(config.discount * (config.T - method.test_time) ) * np.power(r_range_final[0], config.p - 2)
            xaxis0 = states0 
            controls0 = states0 * solution.pi(method.test_time, ys) 
        
            fig, axes = plt.subplots(2, 2, figsize = (12, 12))
            if primal:
                axes[0, 0].plot(PINNx, PINNpi, label = 'Primal PINN, $\\bar{U}$')
            if primal_nonconcave:
                axes[0, 0].plot(PINNx_nonconcave, PINNpi_nonconcave, label = 'Primal PINN, $U$')
            if primal_general:
                axes[0, 0].plot(PINNx_general[:-1], PINNpi_general[:-1], label = 'Primal PINN, $\\bar{U}$ (gen)')
            if primal_general_nonconcave:
                axes[0, 0].plot(PINNx_general_nonconcave[:-1], PINNpi_general_nonconcave[:-1], label = 'Primal PINN, $U$ (gen)')
            if dual:
                axes[0, 0].plot(Dualx, Dualpi, label = 'Dual PINN')
            if dual_general:
                axes[0, 0].plot(Dualx_general[:-1], Dualpi_general[:-1], label = 'Dual PINN (gen)')
            # axes[0, 0].plot(BSDEx, BSDEpi, label = 'BSDE')
            if smp:
                axes[0, 0].plot(SMPx[:-1], SMPpi[:-1], label = 'SMP')
            # if smp_nonconcave:
            #     axes[0, 0].plot(SMPx_nonconcave, SMPpi_nonconcave, label = 'SMP, $U$')
            
            if soln:
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
                axes[0, 1].plot(PINNx_general[:-1], PINNM_general[:-1], label = 'Primal PINN, $\\bar{U}$ (gen)')
            if primal_general_nonconcave:
                axes[0, 1].plot(PINNx_general_nonconcave[:-1], PINNM_general_nonconcave[:-1], label = 'Primal PINN, $U$ (gen)')
            if dual:
                axes[0, 1].plot(Dualx, DualM, label = 'Dual PINN')
            if dual_general:
                axes[0, 1].plot(Dualx_general[:-1], DualM_general[:-1], label = 'Dual PINN (gen)')
            if smp:
                axes[0, 1].plot(monteX_x[:-1], SMPM[:-1], label = 'SMP')
            # if smp_nonconcave:
            #     axes[0, 1].plot(monteX_nonconcave, SMPM_nonconcave, label = 'SMP, $U$')
                
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
                axes[1, 0].plot(PINNx_general[:-1], PINNMx_general[:-1], label = 'Primal PINN, $\\bar{U}$ (gen)')
            if primal_general_nonconcave:
                axes[1, 0].plot(PINNx_general_nonconcave[:-1], PINNMx_general_nonconcave[:-1], label = 'Primal PINN, $U$ (gen)')
            if dual:
                axes[1, 0].plot(Dualx, DualMx, label = 'Dual PINN')
            if dual_general:
                axes[1, 0].plot(Dualx_general[:-1], DualMx_general[:-1], label = 'Dual PINN (gen)')
            if smp:
                axes[1, 0].plot(SMPx[:-1], SMPMx[:-1], label = 'SMP')
            # if smp_nonconcave:
            #     axes[1, 0].plot(SMPx_nonconcave, SMPMx_nonconcave, label = 'SMP, $U$')
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
            if smp:
                SMPmask2 = (SMPt < config.T - 1e-6) & (SMPMxx > threshold) & (SMPMxx < 1.0)
        
            axes[1, 1].set_title('$\partial_{xx}v$ vs x')          
            axes[1, 1].grid(0.5)
            if primal:
                axes[1, 1].plot(PINNx[PINNmask2], PINNMxx[PINNmask2], label = 'Primal PINN, $\\bar{U}$')
            if primal_nonconcave:
                axes[1, 1].plot(PINNx_nonconcave[PINNmask2_nonconcave], PINNMxx_nonconcave[PINNmask2_nonconcave], label = 'Primal PINN, $U$')
            if primal_general:
                axes[1, 1].plot(PINNx_general[PINNmask2_general][:-1], PINNMxx_general[PINNmask2_general][:-1], label = 'Primal PINN, $\\bar{U}$ (gen)')
            if primal_general_nonconcave:
                axes[1, 1].plot(PINNx_general_nonconcave[PINNmask2_general_nonconcave][:-1], PINNMxx_general_nonconcave[PINNmask2_general_nonconcave][:-1], label = 'Primal PINN, $U$ (gen)')
            if dual:
                axes[1, 1].plot(Dualx[Dualmask2], DualMxx[Dualmask2], label = 'Dual PINN')
            if dual_general:
                axes[1, 1].plot(Dualx_general[Dualmask2_general][:-1], DualMxx_general[Dualmask2_general][:-1], label = 'Dual PINN (gen)')
            if smp:
                axes[1, 1].plot(SMPx[SMPmask2][:-1], SMPMxx[SMPmask2][:-1], label = 'SMP')
            if  soln:
                axes[1, 1].plot(states0[derivs20 > 0.01], -1 / derivs20[derivs20 > 0.01], label = 'Reference', c = 'r', linestyle = 'dashed')
            axes[1, 1].set_xlabel('$x$')
            axes[1, 1].set_ylabel('$\partial_{xx}v(0, x, 1)$')
            axes[1, 1].legend()
            
            
            if soln and len(xaxis0) > 0:
                solF = lambda x: np.interp(x, xaxis0[::-1], values0[::-1])
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
                    plt.plot(monteX_x[:-1], SMPM[:-1] - solF(monteX_x[:-1]), label = 'SMP')
                # if smp_nonconcave:
                #     plt.plot(monteX_nonconcave, monteM_nonconcave - solF(monteX_nonconcave), label = 'SMP, $U$')
                plt.plot(xaxis0, np.array(values0) - solF(xaxis0), label = 'Solution', c = 'r', linestyle = 'dotted')
                plt.xlabel('$x$')
                plt.ylabel('Error')
                plt.grid(0.5)
                plt.legend()
                plt.show()   
                
                    
                
        
        # R plots
        if x_range_final[0] <= 1.0 and x_range_final[1] >= 1.0 and r_range_final[0] != r_range_final[1]:
        
            x = 1.0
            
            raxis = np.linspace(config.r_range_final[0], config.r_range_final[1], method.sample_size)
            
        
            
            y = np.array([solution.dual_state(method.test_time, x / r) for r in raxis])
            
            derivs0 = y
                        
            values0 = np.array([solution.value(method.test_time, x / r) for r in raxis]) * np.exp(config.discount * (config.T - method.test_time) ) * np.power(raxis, config.p)
            
            derivs20 = (-1 / solution.d2_v(method.test_time, y)) 
        
            controls0 = - (config.theta_bar * raxis * derivs0) / (config.sigma * x *  derivs20) + config.rho * config.b / config.sigma
        
            
            fig, axes = plt.subplots(1, 2, figsize = (12, 6))
            if primal_general:
                axes[0].plot(PINNr_general_r, PINNpi_general_r, label = 'Primal PINN, $\\bar{U}$ (gen)')
            if primal_general_nonconcave:
                axes[0,].plot(PINNr_general_nonconcave_r, PINNpi_general_nonconcave_r, label = 'Primal PINN, $U$ (gen)')
            if dual_general:
                axes[0].plot(Dualr_general_r, Dualpi_general_r, label = 'Dual PINN (gen)')
            if smp:
                axes[0].plot(SMPr_r[1:], SMPpi_r[1:], label = 'SMP')
            
            if  soln:
                axes[0].plot(raxis, controls0, label = 'Solution', c = 'r', linestyle = 'dashed')
        
            axes[0].set_xlabel('$r$')
            axes[0].set_ylabel('$\pi(0, 1, r)$')
        
            axes[0].set_title('control vs r')
            axes[0].grid(0.5)
            axes[0].legend()
            
            axes[1].set_title('$v$ vs r')    
            axes[1].grid(0.5)
            if primal_general:
                axes[1].plot(PINNr_general_r, PINNM_general_r, label = 'Primal PINN, $\\bar{U}$ (gen)')
            if primal_general_nonconcave:
                axes[1].plot(PINNr_general_nonconcave_r, PINNM_general_nonconcave_r, label = 'Primal PINN, $U$ (gen)')
            if dual_general:
                axes[1].plot(Dualr_general_r, DualM_general_r, label = 'Dual PINN (gen)')
            if smp:
                axes[1].plot(monteR_r[1:], SMPM_r[1:], label = 'SMP')
        
            if soln:
                axes[1].plot(raxis, values0, label = 'Solution', c = 'r', linestyle = 'dashed')
            axes[1].set_xlabel('$r$')
            axes[1].set_ylabel('$v(0, 1, r)$')
            axes[1].legend()
            plt.show()
    
                    
                    
     
                        

    
    
if __name__ == '__main__':
    main()
    
    