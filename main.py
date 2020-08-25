# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 14:41:40 2020

@author: gmvan
"""

import json
import os
import numpy as np
import mvdk
import reader
import matplotlib.pyplot as plt

config = {}


### Parameters of computation
n = 3
config['SAMPLE_DIMENSION'] = [n, n, n] #1; number of nodes for each side (x, y, z)

config['TIME_STEP'] = 5000 # s; #to be automatized # ICI
# can be authomatized as adimensionnal param1 * biggest_void_radius 
config['DELTA_X'] = 2*240E-6 # m; Side length of 1 element
config['NBR_STEPS'] = int(1e7*24*3600/config['TIME_STEP']) # 1; Number of time steps
config['DIMENSION'] = 3 # 1;

### Parameters of system
config['D_EFF'] = 1.5E-12 # m^2/s; Diffusivity of air in saturated cement paste
config['POROSITY'] = 0.3 # 1;
config['ATM_PRESSURE'] = 101325 # Pa;
config['K_H'] = 0.020 / config['ATM_PRESSURE'] # kg/m^3/Pa; Henry's constant
config['CONTACT_ANGLE'] = 0 # rad; Keep equal to 0: code not working for other values
config['TEMPERATURE'] = 293.15 # K;

config['NBR_AIR_VOIDS'] = n # 1; Can presently be equal only to 0 or 1

config['RDM_SEED'] = 0
np.random.seed(0)

for i in range(n):
    r = np.random.normal(200, 15, n**3)
    for idx, i in enumerate(r):
        if i < 160:
            r[idx] = 160
        elif i > 240:
            r[idx] = 240

config['RADIUS_OF_AIR_VOIDS'] = list(r*1e-6) # m;
# config['RADIUS_OF_AIR_VOIDS'] = list(np.ones(n)*100e-6)

config['INDEX_VOIDS'] = [i for i in range(n**3)]

config['BOUNDARY_CONDITION'] = (
    'NO_SIDE_FLOW') # every orthogonal flow null except the upper one (z_max)
    #'NO_FLOW')      # every orthogonal flow considered null
    #'ALL_FLOW')      # all orthogonal flow non null, specimen fully immerged
config['ADIM_PARAM1'] = 0
config['ADIM_PARAM2'] = 1

### Nothing to change below this line
config['SURF_TENSION'] = 70E-3 # J/m^2; Surface tension of air-water interface
config['IDEAL_GAS_CONSTANT'] = 8.314 # J/K/mol;
config['MOLAR_MASS_OF_AIR'] = 28.97E-3 # kg/mol;


# os.makedirs('cas_base_sans_gamma/', exist_ok=True)
# with open('cas_base_sans_gamma/init.json', 'w') as f:
#     json.dump(config, f)    

def test_cube_ten_time():
    seed_name = np.arange(10)
    for idx, i in enumerate(seed_name):
        config['RDM_SEED'] = float(i)
        np.random.seed(i)
        r = np.random.normal(200, 15, n**3)
        for idx, j in enumerate(r):
            if j < 160:
                r[idx] = 160
            elif j > 240:
                r[idx] = 240
        config['RADIUS_OF_AIR_VOIDS'] = list(r*1e-6)
        os.makedirs('test_depth_10_seed_{}'.format(i), exist_ok=True)
        with open('test_depth_10_seed_{}/init.json'.format(i), 'w') as f:
            json.dump(config, f)
        launch_calculation('test_depth_10_seed_{}/'.format(i))


def launch_calculation(path):
    # path being the path to the json init file
    mvdk.launch(path)    

def test_one():
    path_name = np.array([0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 7.5, 10, 25, 50, 75, 100])
    radius_size_test = np.array([0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 7.5, 10, 25, 50, 75, 100])*1e-6
    for i in range(len(path_name)):
        config['RADIUS_OF_AIR_VOIDS'] = radius_size_test[i]
        # as is time_step not optimized (last and first results convergence not assured)
        config['TIME_STEP'] = 20 * (radius_size_test[i]/25e-6)**3
        config['DELTA_X'] = 2.525*radius_size_test[i]
        os.makedirs('test_radius_{}'.format(path_name[i]), exist_ok=True)
        with open('test_radius_{}/init.json'.format(path_name[i]), 'w') as f:
            json.dump(config, f)
        launch_calculation('test_radius_{}/'.format(path_name[i]))

def collapse_time(path):
    f = open(path + 'time.csv', 'r')
    content = f.readlines()
    time2 = []
    for i in content:
        time2.append(float(i))
    f.close()
    return time2[-1]

def main(path):
    os.makedirs(path, exist_ok=True)
    with open(path +'init.json', 'w') as f:
        json.dump(config, f)    
    launch_calculation(path)    


# test_cube_ten_time()
