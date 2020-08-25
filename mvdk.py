# -*- coding: utf-8 -*-
"""
Started 27 April 2020
Solves the saturation of multiple air voids, according to the MVDK model of
Scott Smith
3D only
"""

import numpy as np
import json
import sys
import csv
from scipy.optimize import fsolve
from calculate_mass_matrix import calculate_mass_matrix



class State:
    def __init__(self, path):
        self.param = System_Parameters(path)
        # concentration of dissolved air in liquid water (per unit of v)
        # as m_per_v, NBR_NODES vector representing a (SAMPLE_DIMENSION) table
        base_conc = self.param.BASE_CONC
        self.concentrations = np.ones(self.param.NBR_NODES) * base_conc
        self.masses_per_v = np.ones(self.param.NBR_NODES) * base_conc * self.param.POROSITY
        self.mass_matrix, self.mass_array = calculate_mass_matrix(self.param)
        self.void_sys = Air_Voids_System(self.param)
        self.concentrations, self.masses_per_v = self.void_sys.init_conc(self.concentrations,
                                                                         self.masses_per_v)

    def update_state(self):
        self.masses_per_v += self.param.TIME_STEP * (self.mass_matrix.dot(
                             self.concentrations) + self.mass_array)
        self.concentrations = self.masses_per_v / self.param.POROSITY
        self.concentrations = self.void_sys.update(self.masses_per_v, 
                                                   self.concentrations)


class Air_Voids_System:
    def __init__(self, param):
        self.param = param
        self.number = self.param.NBR_AIR_VOIDS
        self.voids = np.zeros(self.number).astype(Air_Void)
        self.index = self.param.INDEX_VOIDS
        for idx, i in enumerate(self.index):
            self.voids[idx] = Air_Void(self.param, idx, i)    

    def init_conc(self, concentrations, masses_per_v):
        for void in self.voids:
            concentrations[void.index] = void.paste_conc
            masses_per_v[void.index] = void.m_per_v
        return concentrations, masses_per_v

    def update(self, masses_per_v, concentrations):
        for void in self.voids:
            void.update(masses_per_v[void.index])
            concentrations[void.index] = void.paste_conc
        return concentrations
    

class Air_Void:
    # probably a change on param tbd for muti-voids
    # like spefic params of voids
    def __init__(self, param, sys_num, index):
        self.param = param
        self.sys_num = sys_num # notation in the void sys
        self.index = index # notation to find in the conc table
        self.void_rad = self.param.RADIUS_OF_AIR_VOIDS[self.sys_num]
        self.gamma = self.param.GAMMA[self.sys_num]
        self.init_limit_mass()
        self.init_bubble_rad()

    def init_limit_mass(self):
        # determining minimal mass limit
        a = np.linspace(0, self.void_rad, 1001)
        a = a[1:] # tbi
        b = np.zeros(1000)
        for i in range(len(b)):
            self.bub_rad = a[i]
            b[i] = self.avg_mass_per_v_around_bubble()
        self.min_mass_per_v = np.min(b)
        # determing maximal mass limit w/ Rgl = Rv
        mass_bubble = 4/3*np.pi*self.void_rad**3*(
                      self.param.ATM_PRESSURE + 2*self.param.SURF_TENSION/self.void_rad)*(
                      self.param.MOLAR_MASS_OF_AIR/self.param.IDEAL_GAS_CONSTANT
                      /self.param.TEMPERATURE)
        mass_paste = (self.param.DELTA_X**3 - 4/3*np.pi*self.void_rad**3) * (
                      self.param.POROSITY * self.param.K_H*(
                      self.param.ATM_PRESSURE +
                      self.gamma *2*self.param.SURF_TENSION/self.void_rad))
        self.max_mass_per_v = (mass_bubble+mass_paste)/(self.param.DELTA_X**3)

    def init_bubble_rad(self):
        def func_whose_root_gives_initial_bubble_rad(bub_rad, void_rad):
            return (1 - 4/3*np.pi*bub_rad**3
                    * (self.param.ATM_PRESSURE + 2*self.param.SURF_TENSION/bub_rad)
                    /(4/3*np.pi*void_rad**3*self.param.ATM_PRESSURE))
        res = fsolve(func_whose_root_gives_initial_bubble_rad, 0.95*self.void_rad, (self.void_rad))
        self.bub_rad = res[0]
        # specific initialization considering an initial conservation of gazeous air
        mass_air_in_bubble = (self.param.MOLAR_MASS_OF_AIR * 4/3*np.pi*
                self.void_rad**3 * self.param.ATM_PRESSURE / self.param.IDEAL_GAS_CONSTANT
                / self.param.TEMPERATURE)
        self.calculate_conc_if_bubble()
        self.calculate_conc_if_bubble_paste()
        mass_air_around_bubble = ( self.paste_conc *
            self.param.POROSITY * (self.param.DELTA_X**3 - 4/3*np.pi*self.void_rad**3)
            + self.void_conc *4/3*np.pi*(self.void_rad**3 - self.bub_rad**3))
        self.m_per_v = (mass_air_in_bubble + mass_air_around_bubble) / self.param.DELTA_X**3
    
    def update(self, mass_per_v):
        self.m_per_v = mass_per_v
        if self.m_per_v < self.min_mass_per_v or self.bub_rad == 0:
        # respecting low limit and ensuring a bubble can't appear after collapse
            self.bub_rad = 0
            self.calculate_conc_if_no_bubble()
        elif self.m_per_v >= self.max_mass_per_v:
            self.bub_rad = self.void_rad
            self.calculate_conc_if_full_bubble()
        else:
            self.calculate_bub_rad()
            self.calculate_conc_if_bubble_paste()
            self.calculate_conc_if_bubble()
        return self.paste_conc

    def avg_mass_per_v_around_bubble(self):
        self.calculate_conc_if_bubble()
        self.calculate_conc_if_bubble_paste()
        mass_around_void = self.paste_conc * self.param.POROSITY * (
                               self.param.DELTA_X**3 - 4/3*np.pi*self.void_rad**3)
        mass_air_bubble = (self.param.MOLAR_MASS_OF_AIR * (self.param.ATM_PRESSURE
                           + self.param.SURF_TENSION *2/self.bub_rad)
                           / self.param.IDEAL_GAS_CONSTANT / self.param.TEMPERATURE
                           * self.bubble_volume())
        mass_liquid_in_air_void = self.void_conc * 4/3*np.pi*(self.void_rad**3 - self.bub_rad**3)
        return ((mass_around_void + mass_air_bubble + mass_liquid_in_air_void) /
                 self.param.DELTA_X**3)
    
    def calculate_bub_rad(self):
        def func_whose_root_gives_radius(bub_rad, m):
            self.bub_rad = bub_rad
            total_mass_wanted = m
            total_mass = self.avg_mass_per_v_around_bubble()
            return total_mass_wanted - total_mass
        res = fsolve(func_whose_root_gives_radius, 0.5*self.void_rad, (self.m_per_v))
        self.bub_rad = res[0]
            
    
    def calculate_conc_if_bubble(self):
        courb = 2/self.bub_rad
        self.void_conc = self.param.K_H * (self.param.ATM_PRESSURE
                              + self.param.SURF_TENSION * courb)
    
    def calculate_conc_if_bubble_paste(self):
        courb = 2/self.bub_rad
        self.paste_conc = self.param.K_H * (self.param.ATM_PRESSURE
                              + self.gamma * self.param.SURF_TENSION * courb)
        
    def calculate_conc_if_full_bubble(self):
        courb = 2/self.void_rad
        over_pressure = (self.m_per_v - self.max_mass_per_v)*self.param.DELTA_X**3/(
            4/3*np.pi*self.void_rad**3*self.param.MOLAR_MASS_OF_AIR/self.param.TEMPERATURE/self.param.IDEAL_GAS_CONSTANT + (
            self.param.K_H * self.gamma* self.param.POROSITY * (self.param.DELTA_X**3 - 4/3*np.pi*self.void_rad**3)))
        self.paste_conc = self.param.K_H * (self.param.ATM_PRESSURE
                          + self.gamma * (self.param.SURF_TENSION * courb + over_pressure))
        
        
    def calculate_conc_if_no_bubble(self):
        V = self.param.DELTA_X**3
        V_sphere = 4/3 * np.pi * self.void_rad**3
        self.void_conc = self.m_per_v * V/(V_sphere + self.param.POROSITY * (V - V_sphere))
        self.paste_conc = self.m_per_v * V/(V_sphere + self.param.POROSITY * (V - V_sphere))

    def bubble_volume(self): # returns volume (in m^3) of
        # air bubble of radius 'bub_rad', in air void of radius 'void_rad'.
        # Uses 'CONTACT_ANGLE'
        return 4/3 * np.pi * self.bub_rad**3
    #    # Scott's formula
    #    return (np.pi / 12) * (1 / np.sqrt(void_rad**2 + bub_rad**2 - 2*void_rad*
    #           bub_rad*np.cos(CONTACT_ANGLE))) * ((void_rad + bub_rad - np.sqrt(
    #                   void_rad**2 + bub_rad**2 - 2*void_rad*bub_rad*
    #                   np.cos(CONTACT_ANGLE)))**2) * ((void_rad**2 + bub_rad**2 -
    #                   2*void_rad*bub_rad*np.cos(CONTACT_ANGLE)) + (2*np.sqrt(
    #                           void_rad**2 + bub_rad**2 - 2*void_rad*bub_rad*
    #                           np.cos(CONTACT_ANGLE))*(void_rad + bub_rad)) - 3*(
    #                           void_rad**2 + bub_rad**2) + 6*void_rad*bub_rad)


class Data_Storage: # defines what data to store
    def __init__(self, state): # to be initiated with the initial state 'state'
        self.time = np.array(0.)
        # self.concentrations = state.concentrations
        # self.masses_per_v = state.masses_per_v
        # self.void_conc = state.concentrations[state.void_sys.voids[0].index]
        npx = state.param.SAMPLE_DIMENSION[0]
        npy = state.param.SAMPLE_DIMENSION[1]
        a = np.arange(state.param.NBR_NODES).reshape((state.param.SAMPLE_DIMENSION))
        self.cross_section_index = a[int(npx/2), int(npy/2), :]
        self.section_conc = state.concentrations[self.cross_section_index]
        self.rad_evol = np.array([void.bub_rad for void in state.void_sys.voids])   
        self.conc = state.concentrations

    # def store_all(self, t, state): # stores data from state 'state' at time 't'
    #     #tbu
    #     self.time = np.concatenate((self.time, np.array([t])), 0)
    #     c = state.concentrations.reshape((NUMBER_OF_NODES, 1))
    #     self.concentrations = np.concatenate((self.concentrations, c), 1)
    #     m = state.masses_per_volume.reshape((NUMBER_OF_NODES, 1))
    #     self.masses_per_volume = np.concatenate((self.masses_per_volume, m), 1)
    #     if (NUMBER_OF_AIR_VOIDS == 1):
    #         self.bubble_radius = np.concatenate(
    #                 (self.bubble_radius, state.void_sys.bubbles_radius), 0)
            
    
    def store_rad(self, t, state):
        self.time = np.vstack((np.array(self.time), np.array([t])))
        # current_void_conc = state.concentrations[state.void_sys.voids[0].index]
        # self.void_conc = np.vstack((self.void_conc,
        #                                     current_void_conc))
        self.rad_evol = np.vstack((self.rad_evol, 
                                   np.array([void.bub_rad for void in state.void_sys.voids])))
    
    def store_rad_section_conc(self, t, state):
        self.time = np.vstack((np.array(self.time), np.array([t])))
        self.rad_evol = np.vstack((self.rad_evol, 
                                   np.array([void.bub_rad for void in state.void_sys.voids])))
        self.section_conc = np.vstack((self.conc,
                                       state.concentrations[self.cross_section_index]))
    
    def store_rad_conc(self, t, state):
        self.time = np.vstack((np.array(self.time), np.array([t])))
        self.rad_evol = np.vstack((self.rad_evol, 
                                   np.array([void.bub_rad for void in state.void_sys.voids])))
        self.conc = np.vstack((self.conc,
                                       np.array(state.concentrations)))
        


class System_Parameters:
    def __init__(self, path):
        with open(path, 'r') as f:
            config = json.load(f)

        self.SAMPLE_DIMENSION = np.array(config['SAMPLE_DIMENSION'])
        self.NBR_NODES = self.SAMPLE_DIMENSION.prod()
        
        self.TIME_STEP = config['TIME_STEP']
        self.DELTA_X = config['DELTA_X']
        self.NBR_STEPS = config['NBR_STEPS']
        
        self.D_EFF = config['D_EFF']
        self.POROSITY = config['POROSITY']
        self.ATM_PRESSURE = config['ATM_PRESSURE']
        self.K_H = config['K_H']
        self.CONTACT_ANGLE = config['CONTACT_ANGLE']
        self.TEMPERATURE = config['TEMPERATURE']
        self.SURF_TENSION = config['SURF_TENSION']
        
        self.INDEX_VOIDS = np.array(config['INDEX_VOIDS']).astype(int)
        self.NBR_AIR_VOIDS = len(self.INDEX_VOIDS)
        self.RADIUS_OF_AIR_VOIDS = config['RADIUS_OF_AIR_VOIDS']
        
        self.DIMENSION = config['DIMENSION']
        self.BOUNDARY_CONDITION = config['BOUNDARY_CONDITION']
        self.ADIM_PARAM1 = config['ADIM_PARAM1']
        self.ADIM_PARAM2 = config['ADIM_PARAM2']
        self.GAMMA = ((np.array(self.RADIUS_OF_AIR_VOIDS)/self.DELTA_X)
                       **self.ADIM_PARAM1 * self.ADIM_PARAM2)
        
        self.IDEAL_GAS_CONSTANT = config['IDEAL_GAS_CONSTANT']
        self.MOLAR_MASS_OF_AIR = config['MOLAR_MASS_OF_AIR']
        
        # specific variable to exterior conc test
        if 'EXT_CONC_PERCENT' in config:
            self.BASE_CONC = (self.K_H * self.ATM_PRESSURE
                              * (1 + config['EXT_CONC_PERCENT']/100))
        else:
            self.BASE_CONC = self.K_H * self.ATM_PRESSURE
        
        self.TIMES = np.arange(self.NBR_STEPS + 1) * self.TIME_STEP
        # verification
        if (2*self.RADIUS_OF_AIR_VOIDS[0] > self.DELTA_X): # CAREFUL ABOUT THAT
            print('The radius of the air voids is too large for the grid')
            sys.exit()

def launch(path):
    current_state = State(path + 'init.json')
    data_history = Data_Storage(current_state)
    day = 0
    i = 0
    dt = current_state.param.TIME_STEP
    void_verif = np.array([void.bub_rad for void in current_state.void_sys.voids])
    while max(void_verif) > 0:
        i += 1
        current_state.update_state()
        if i%8 == 0:
            data_history.store_rad_conc(i*dt, current_state)
        void_verif = np.array([void.bub_rad for void in current_state.void_sys.voids])
        if i*dt > 2000*day*86400:
            data_history.store_rad_conc(i*dt, current_state)
            np.savetxt(path + 'conc.csv', data_history.conc, delimiter=',', fmt='%1.4e')
            np.savetxt(path + 'bubble_radius.csv', data_history.rad_evol, delimiter=',', fmt='%1.4e')
            np.savetxt(path + 'time.csv', data_history.time, delimiter=',', fmt='%1.4e')
            print(void_verif, 2000*day)
            day += 1
            
    
    #to be put in the data_history class
    np.savetxt(path + 'bubble_radius.csv', data_history.rad_evol, delimiter=',', fmt='%1.4e')
    np.savetxt(path + 'conc.csv', data_history.conc, delimiter=',', fmt='%1.4e')
    np.savetxt(path + 'time.csv', data_history.time, delimiter=',', fmt='%1.4e')
    return current_state
    
def main():
    launch('250_test_dx_10/')


if __name__ == '__main__':
    main()
    
    
    
            # if (self.number == 1):
        #     # we consider a central node, determining here its index
        #     a = np.arange(self.param.NBR_NODES)
        #     a = a.reshape((self.param.SAMPLE_DIMENSION))
        #     center_idx = (self.param.SAMPLE_DIMENSION/2).astype(int)
        #     center_node = a[center_idx[0], center_idx[1], center_idx[2]]
        #     self.index[0] = center_node
        #     self.voids[0] = Air_Void(self.param, self.param.RADIUS_OF_AIR_VOIDS, self.param.GAMMA, self.index[0])
        # if self.number == 2:
        #     a = np.arange(self.param.NBR_NODES)
        #     a = a.reshape((self.param.SAMPLE_DIMENSION))
        #     first_idx = (self.param.SAMPLE_DIMENSION/2).astype(int) + (0, 0, -1)
        #     sec_idx = (self.param.SAMPLE_DIMENSION/2).astype(int) + (0, 0, +1)
        #     first_node = a[first_idx[0], first_idx[1], first_idx[2]]
        #     self.index[0] = first_node
        #     # insert an intern to air voids system void index
        #     self.voids[0] = Air_Void(self.param, self.param.RADIUS_OF_AIR_VOIDS[0], self.param.GAMMA[0], self.index[0])
        #     second_node = a[sec_idx[0], sec_idx[1], sec_idx[2]]
        #     self.index[1] = second_node
        #     self.voids[1] = Air_Void(self.param, self.param.RADIUS_OF_AIR_VOIDS[1], self.param.GAMMA[1], self.index[1])