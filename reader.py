# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 14:29:42 2020

@author: gmvan
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from mvdk import System_Parameters

def read_rad_evol(path):
    f = open(path + 'bubble_radius.csv', "r")
    content = f.readlines()
    n = len(content[0].split(','))
    radius_evolution = np.zeros((len(content), n))
    for idx, r_list in enumerate(content):
        liste_r = r_list.split(',')
        for i in range(n):
            if float(liste_r[i]) == 0.:
                break
            radius_evolution[idx, i] = float(liste_r[i])
    f.close()
    time = []
    f = open(path + 'time.csv', 'r')
    content = f.readlines()
    for i in content:
        time.append(float(i))
    time = np.array(time)
    for i in range(len(radius_evolution[0])):
        radius_evol = radius_evolution[:, i]
        if 0. in radius_evol:
            radius_evol = radius_evol[:list(radius_evol).index(0.)]
            np.array(list(radius_evol).append(0.))
        plt.plot(time[:len(radius_evol)]/24/3600, 1e6*radius_evol, label='{} node'.format(i + 1))
    plt.xlabel('time (day)')
    plt.ylabel('Radius of air bubble (um)')
    plt.savefig('fig_cube_test.pdf')
    # plt.legend(loc='lower left')
    f.close()



def read_radius_evol(path, SVDK = 0):
    if SVDK != 0:
        if SVDK == 25:
            f = open(path + "raw_data_fig17\\Figure17_100um.txt", "r")
            content = f.readlines()
            radius_evolution = []
            time = []
            for i in content[1:]:
                a = i.split()
                radius_evolution.append(float(a[1]))
                time.append(float(a[0]))
            time = np.array(time)/3600/24
            radius_evolution = np.array(radius_evolution)
            plt.plot(time, 1e6*radius_evolution, label='SVDK model')
            f.close()
    
    f = open(path + 'bubble_radius.csv', "r")
    content = f.readlines()
    radius_evolution2 = []
    time2 = []
    for i in content:
        radius_evolution2.append(float(i))
    
    f.close()
    f = open(path + 'time.csv', 'r')
    content = f.readlines()
    for i in content:
        time2.append(float(i))
        
    radius_evolution2 = np.array(radius_evolution2)
    
    
    plt.plot(np.array(time2)/24/3600, 1e6*radius_evolution2, label='MVDK model')
    plt.xlabel('time (day)')
    plt.ylabel('Radius of air bubble (um)')
    plt.legend()
    f.close()
    
def read_last_conc(path):
    f = open(path + 'current_concentration.csv', "r")
    content = f.readlines()
    current_concentration = np.zeros(len(content))
    for idx, i in enumerate(content):
        current_concentration[idx] = i
    f.close()
    
    parameters = System_Parameters(path + '/init.json')
    
    
    n = parameters.NBR_NODES_PER_SIDE
    DIMENSION = parameters.DIMENSION
    if DIMENSION == 3:
        indices = np.arange(np.power(n, 3)).reshape((n,n,n))
        indices_middle = indices[int(n/2), int(n/2), :]
    concentration_middle = [current_concentration[i] for i in indices_middle]
    z_scale = np.arange(n)*parameters.DELTA_X
    
    plt.figure()
    plt.plot(z_scale, concentration_middle, 'o')
    plt.plot([0.00125 - 0.00016110060506910545], [1.02*parameters.K_H*parameters.ATM_PRESSURE], 'o')
    plt.xlabel('Depth (m)')
    plt.ylabel('Concentration (kg/m^3)' )
    
    

def read_all_r_evol(path):
    os.chdir(path)
    for direct in os.scandir():
        if direct.is_dir():
            f = open(direct.path[2:] + '\\' + 'bubble_radius.csv', "r")
            content = f.readlines()
            radius_evolution2 = []
            time2 = []
            for i in content:
                radius_evolution2.append(float(i))
            f.close()
            f = open(direct.path[2:] + '\\' + 'time.csv', 'r')
            content = f.readlines()
            for i in content:
                time2.append(float(i))
            radius_evolution2 = np.array(radius_evolution2)
            plt.plot(np.array(time2)/24/3600, 1e6*radius_evolution2, label=direct.path)
            plt.xlabel('time (day)')
            plt.ylabel('Radius of air bubble (um)')
            plt.legend(fontsize='small', loc='upper right')