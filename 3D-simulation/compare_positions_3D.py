#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 12:56:58 2018

@author: root
"""

import wsn_lite_webots
import matplotlib.pyplot as plt
import numpy as np

X_GRID = 16
Y_GRID = 5
Z_GRID = 3

# Time
delta = 5.0  # minutes
tc = 0  # minutes

# Parameters
half_life = 0.01  # seconds DEFAULT VALUE IS 0.25 [s]
bouts_amp_thresh = 1000  # DEFAULT VALUE IS 0.13

W1 = wsn_lite_webots.wsn('3D_spx15.0_spy2.0_spz0.1_cluttered')
W2 = wsn_lite_webots.wsn('3D_spx15.0_spy2.0_spz0.1_ws_0.75')
W3 = wsn_lite_webots.wsn('3D_spx15.0_spy2.0_spz0.2_cluttered')
W4 = wsn_lite_webots.wsn('3D_spx15.0_spy2.0_spz0.2_ws_0.75')
W5 = wsn_lite_webots.wsn('3D_spx15.0_spy2.0_spz0.3_cluttered')
W6 = wsn_lite_webots.wsn('3D_spx15.0_spy2.0_spz0.3_ws_0.75')
list_W = [W1, W2, W3, W4, W5, W6]
list_bouts_amp_thresh = [None]*len(list_W)

TSL1 = [0.1, 2, 15]  # (z, y, x)
TSL2 = [0.1, 2, 15]  # (z, y, x)
TSL3 = [0.2, 2, 15]  # (z, y, x)
TSL4 = [0.2, 2, 15]  # (z, y, x)
TSL5 = [0.3, 2, 15]  # (z, y, x)
TSL6 = [0.3, 2, 15]  # (z, y, x)

list_TSL = [TSL1, TSL2, TSL3, TSL4, TSL5, TSL6]

# plt.figure()
# for i in range(180, W.c_ppm.shape[1]):
#     plt.plot(W.t_s, W.c_ppm[:, i], label=f'({i//Y_GRID}, {i%Y_GRID}, 0)')
# plt.legend(ncol=2)
# plt.xlabel('time [s]')
# plt.ylabel('concentration [ppm]')
# plt.title('Concentration of 75 sensors')

for i in range(len(list_W)):
    list_W[i].plotGasMap(map_type='mean', timeframe=[tc-delta/2, tc+delta/2], tsl=list_TSL[i])

    list_bouts_amp_thresh[i] = list_W[i].compute_bouts_amps_threshold(timeframe=[tc - delta / 2, tc + delta / 2],
                                                                      hl=half_life, method='knee', plot=False)
    list_W[i].plotGasMap(map_type='bouts-freq', timeframe=[tc-delta/2, tc+delta/2], bouts_hl=half_life,
                 bouts_ampthresh=list_bouts_amp_thresh[i], tsl=list_TSL[i])

plt.show()
