#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 12:56:58 2018

@author: root
"""

import wsn_lite_webots
import matplotlib.pyplot as plt
import numpy as np

X_GRID = 29
Y_GRID = 7
Z_GRID = 1

# Time
delta = 5.0  # minutes
tc = 0  # minutes

# Parameters
half_life = 0.01  # seconds DEFAULT VALUE IS 0.25 [s]
bouts_amp_thresh = 5e3  # DEFAULT VALUE IS 0.13

W = wsn_lite_webots.wsn('2D_cluttered')
TSL = [0.1, 2, 15]  # (z, y, x)

print(W.c_ppm.shape, W.t_s.shape)
plt.figure()
for i in range(180, W.c_ppm.shape[1]):
    plt.plot(W.t_s, W.c_ppm[:, i], label=f'({i//Y_GRID}, {i%Y_GRID}, 0)')
plt.legend(ncol=2)
plt.xlabel('time [s]')
plt.ylabel('concentration [ppm]')
plt.title('Concentration of 75 sensors')

W.plotGasMap(map_type='mean', timeframe=[tc-delta/2, tc+delta/2], tsl=TSL)
W.plotGasMap(map_type='bouts-freq', timeframe=[tc-delta/2, tc+delta/2], bouts_hl=half_life,
             bouts_ampthresh=bouts_amp_thresh, tsl=TSL)

bouts_amp_thresh = W.compute_bouts_amps_threshold(timeframe=[tc-delta/2, tc+delta/2], hl=half_life)
print(bouts_amp_thresh)
plt.show()
