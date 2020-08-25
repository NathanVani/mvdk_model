# -*- coding: utf-8 -*-
"""
Started 27 April 2020
Calculate mass matrix necessary to solve MVDK system
through a Euler method
"""

import numpy as np
from scipy.sparse import lil_matrix


def calculate_mass_matrix(param):
    n = param.NBR_NODES
    sample_shape = param.SAMPLE_DIMENSION
    nx, ny, nz = sample_shape[0], sample_shape[1], sample_shape[2]
    nyz = ny * nz # allows to travel on the x axis
    DIMENSION = param.DIMENSION
    BOUNDARY_CONDITION = param.BOUNDARY_CONDITION
    D_EFF = param.D_EFF
    DELTA_X = param.DELTA_X
    # where 'n' is the number of nodes per side
    m = lil_matrix((n, n))
    b = np.zeros(n)
    concentration_ext = param.BASE_CONC #K_H * ATM_PRESSURE * (1 + param.EXT_CONC_PERCENT/100) # kg/m^3 ; air concentration
    # in liquid water at the surface
    if DIMENSION == 3:
        indices = np.arange(n).reshape(sample_shape)
        # We identify the indices of the nodes on the various faces
        indices_x_min = indices[0,:,:].reshape(ny*nz)
        indices_x_max = indices[-1,:,:].reshape(ny*nz)
        indices_y_min = indices[:,0,:].reshape(nx*nz)
        indices_y_max = indices[:,-1,:].reshape(nx*nz)
        indices_z_min = indices[:,:,0].reshape(nx*ny)
        indices_z_max = indices[:,:,-1].reshape(nx*ny)
        # We identify the indices of the nodes that are not on the faces
        indices_bulk = np.setdiff1d(indices, np.concatenate((
                           indices_x_min, indices_x_max,
                           indices_y_min, indices_y_max,
                           indices_z_min, indices_z_max)))
        for i in indices_bulk:
            m[i, i] = -6
            m[i, i + 1] = 1
            m[i, i - 1] = 1
            m[i, i + nz] = 1
            m[i, i - nz] = 1
            m[i, i + nyz] = 1
            m[i, i - nyz] = 1
        if BOUNDARY_CONDITION == 'ALL_FLOW':
            # all surfaces are considered in contact with water, thus
            # taking into account a specific flow with a fixed concentration
            indices_boundary = np.unique(np.concatenate((
                                       indices_x_min, indices_x_max,
                                       indices_y_min, indices_y_max,
                                       indices_z_min, indices_z_max)))
            for i in indices_boundary:
                m[i, i] = -6
                if i not in indices_x_min:
                    m[i, i - nyz] = 1
                else:
                    m[i, i] -= 1
                    b[i] += 2 * concentration_ext
                if i not in indices_x_max:
                    m[i, i + nyz] = 1
                else:
                    m[i, i] -= 1
                    b[i] += 2 * concentration_ext
                if i not in indices_y_min:
                    m[i, i - nz] = 1
                else:
                    m[i, i] -= 1
                    b[i] += 2 * concentration_ext
                if i not in indices_y_max:
                    m[i, i + nz] = 1
                else:
                    m[i, i] -= 1
                    b[i] += 2 * concentration_ext
                if i not in indices_z_min:
                    m[i, i - 1] = 1
                else:
                    m[i, i] -= 1
                    b[i] += 2 * concentration_ext
                if i not in indices_z_max:
                    m[i, i + 1] = 1
                else:
                    m[i, i] -= 1
                    b[i] += 2 * concentration_ext
        if BOUNDARY_CONDITION == 'NO_SIDE_FLOW':
            # only upper (z_max) surface considered in contact with water,
            # 5 others considered with no flow perpendicular to the surface
            indices_boundary = np.unique(np.concatenate((
                               indices_x_min, indices_x_max,
                               indices_y_min, indices_y_max,
                               indices_z_min, indices_z_max)))
            for i in indices_boundary:
                m[i, i] = -6
                if i not in indices_x_min:
                    m[i, i - nyz] = 1
                else:
                    m[i, i] += 1
                if i not in indices_x_max:
                    m[i, i + nyz] = 1
                else:
                    m[i, i] += 1
                if i not in indices_y_min:
                    m[i, i - nz] = 1
                else:
                    m[i, i] += 1
                if i not in indices_y_max:
                    m[i, i + nz] = 1
                else:
                    m[i, i] += 1
                if i not in indices_z_min:
                    m[i, i - 1] = 1
                else:
                    m[i, i] += 1
                if i not in indices_z_max:
                    m[i, i + 1] = 1
                else:
                    m[i, i] -= 1
                    b[i] += 2 * concentration_ext
        if BOUNDARY_CONDITION == 'NO_FLOW':
            # all perpendicular flow to surfaces considered null
            indices_boundary = np.unique(np.concatenate((
                               indices_x_min, indices_x_max,
                               indices_y_min, indices_y_max,
                               indices_z_min, indices_z_max)))
            for i in indices_boundary:
                m[i, i] = -6
                if i not in indices_x_min:
                    m[i, i - nyz] = 1
                else:
                    m[i, i] += 1
                if i not in indices_x_max:
                    m[i, i + nyz] = 1
                else:
                    m[i, i] += 1
                if i not in indices_y_min:
                    m[i, i - nz] = 1
                else:
                    m[i, i] += 1
                if i not in indices_y_max:
                    m[i, i + nz] = 1
                else:
                    m[i, i] += 1
                if i not in indices_z_min:
                    m[i, i - 1] = 1
                else:
                    m[i, i] += 1
                if i not in indices_z_max:
                    m[i, i + 1] = 1
                else:
                    m[i, i] += 1
    # if DIMENSION == 2:
    #     indices = np.arange(np.power(n, 2)).reshape((n,n))
    #     indices_x_min = indices[0,:].reshape(n)
    #     indices_x_max = indices[-1,:].reshape(n)
    #     indices_z_min = indices[:,0].reshape(n)
    #     indices_z_max = indices[:,-1].reshape(n)
    #     indices_bulk = np.setdiff1d(indices, 
    #                    np.concatenate((indices_x_min, indices_x_max,
    #                                    indices_z_min, indices_z_max)))
    #     for i in indices_bulk:
    #         m[i, i] = -4
    #         m[i, i + 1] = 1
    #         m[i, i - 1] = 1
    #         m[i, i + n] = 1
    #         m[i, i - n] = 1
    #     if BOUNDARY_CONDITION == 'ALL_FLOW':
    #         indices_boundary = np.unique(np.concatenate((
    #                                indices_x_min, indices_x_max,
    #                                indices_z_min, indices_z_max)))
    #         for i in indices_boundary:
    #             m[i, i] = -4
    #             if i not in indices_x_min and i not in indices_x_max:
    #                 m[i, i + n] = 1
    #                 m[i, i - n] = 1
    #             if i not in indices_z_min and i not in indices_z_max:
    #                 m[i, i + 1] = 1
    #                 m[i, i - 1] = 1
    #             if i in indices_x_min:
    #                 m[i, i] = m[i, i] - 1
    #                 m[i, i + n] = 1
    #                 b[i] += 2 * concentration_ext
    #             if i in indices_x_max:
    #                 m[i, i] = m[i, i] - 1
    #                 m[i, i - n] = 1
    #                 b[i] += 2 * concentration_ext
    #             if i in indices_z_min:
    #                 m[i, i] = m[i, i] - 1
    #                 m[i, i + 1] = 1
    #                 b[i] += 2 * concentration_ext
    #             if i in indices_z_max:
    #                 m[i, i] = m[i, i] - 1
    #                 m[i, i - 1] = 1
    #                 b[i] += 2 * concentration_ext
    #     if BOUNDARY_CONDITION == 'NO_SIDE_FLOW':
    #         indices_boundary = np.unique(np.concatenate((
    #                                indices_x_min, indices_x_max,
    #                                indices_z_min, indices_z_max)))
    #         for i in indices_boundary:
    #             m[i, i] = -4
    #             if i not in indices_x_min and i not in indices_x_max:
    #                 m[i, i + n] = 1
    #                 m[i, i - n] = 1
    #             if i not in indices_z_min and i not in indices_z_max:
    #                 m[i, i + 1] = 1
    #                 m[i, i - 1] = 1
    #             if i in indices_x_min:
    #                 m[i, i] = m[i, i] + 1
    #                 m[i, i + n] = 1
    #             if i in indices_x_max:
    #                 m[i, i] = m[i, i] + 1
    #                 m[i, i - n] = 1
    #             if i in indices_z_min:
    #                 m[i, i] = m[i, i] + 1
    #                 m[i, i + 1] = 1
    #             if i in indices_z_max:
    #                 m[i, i] = m[i, i] - 1
    #                 m[i, i - 1] = 1
    #                 b[i] += 2 * concentration_ext
    #     if BOUNDARY_CONDITION == 'NO_FLOW':
    #         indices_interior = np.unique(np.concatenate((
    #                                indices_x_min, indices_x_max,
    #                                indices_z_min, indices_z_max)))
    #         for i in indices_interior:
    #             m[i, i] = -4
    #             if i not in indices_x_min and i not in indices_x_max:
    #                 m[i, i + n] = 1
    #                 m[i, i - n] = 1
    #             if i not in indices_z_min and i not in indices_z_max:
    #                 m[i, i + 1] = 1
    #                 m[i, i - 1] = 1
    #             if i in indices_x_min:
    #                 m[i, i] = m[i, i] + 1
    #                 m[i, i + n] = 1
    #             if i in indices_x_max:
    #                 m[i, i] = m[i, i] + 1
    #                 m[i, i - n] = 1
    #             if i in indices_z_min:
    #                 m[i, i] = m[i, i] + 1
    #                 m[i, i + 1] = 1
    #             if i in indices_z_max:
    #                 m[i, i] = m[i, i] + 1
    #                 m[i, i - 1] = 1
    m = m * D_EFF / (np.power(DELTA_X, 2))
    b = b * D_EFF / (np.power(DELTA_X, 2))
    return m, b