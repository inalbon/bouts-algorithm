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

W1 = wsn_lite_webots.wsn('3D_spx12.5_spy3.0_spz1.0_ws_0.75')
W2 = wsn_lite_webots.wsn('3D_spx13.0_spy2.0_spz1.4_ws_0.75')
W3 = wsn_lite_webots.wsn('3D_spx14.0_spy2.0_spz1.2_ws_0.75')
W4 = wsn_lite_webots.wsn('3D_spx14.2_spy2.3_spz1.3_ws_0.75')
W5 = wsn_lite_webots.wsn('3D_spx15.0_spy2.0_spz1.0_ws_0.75')
list_W = [W1, W2, W3, W4, W5]
list_bouts_amp_thresh = []

TSL1 = [1, 3, 12.5]  # (z, y, x)
TSL2 = [1.4, 2, 13]  # (z, y, x)
TSL3 = [1.2, 2, 14]  # (z, y, x)
TSL4 = [1.3, 2.3, 14.2]  # (z, y, x)
TSL5 = [1.0, 2, 15]  # (z, y, x)

list_TSL = [TSL1, TSL2, TSL3, TSL4, TSL5]
error_mean = []
error_bouts = []

# plt.figure()
# for i in range(180, W.c_ppm.shape[1]):
#     plt.plot(W.t_s, W.c_ppm[:, i], label=f'({i//Y_GRID}, {i%Y_GRID}, 0)')
# plt.legend(ncol=2)
# plt.xlabel('time [s]')
# plt.ylabel('concentration [ppm]')
# plt.title('Concentration of 75 sensors')

i = 0
for W in list_W:
    W.plotGasMap(map_type='mean', timeframe=[tc-delta/2, tc+delta/2], tsl=list_TSL[i])

    list_bouts_amp_thresh.append(list_W[i].compute_bouts_amps_threshold(timeframe=[tc - delta / 2, tc + delta / 2],
                                                                        hl=half_life, method='knee', plot=False))
    W.plotGasMap(map_type='bouts-freq', timeframe=[tc-delta/2, tc+delta/2], bouts_hl=half_life,
                 bouts_ampthresh=list_bouts_amp_thresh[i], tsl=list_TSL[i])

    error_mean.append(W.computeError(map_type='mean', timeframe=[tc-delta/2, tc+delta/2], tsl=list_TSL[i]))
    error_bouts.append(W.computeError(map_type='bouts-freq', timeframe=[tc-delta/2, tc+delta/2], tsl=list_TSL[i],
                                      bouts_hl=half_life, bouts_ampthresh=list_bouts_amp_thresh[i]))
    i = i+1

print('mean errors', error_mean)
print('bouts errors', error_bouts)

plt.show()
