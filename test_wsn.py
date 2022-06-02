#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 12:56:58 2018

@author: root
"""

import wsn_lite
import matplotlib.pyplot as plt
import numpy as np

# Time
delta = 5.0  # minutes
tc = 60  # minutes

# Parameters
half_life = 0.25  # seconds DEFAULT VALUE IS 0.25 [s]
bouts_amp_thresh = 0.13  # DEFAULT VALUE IS 0.13

# True source location of 10 experiments
TSL1 = [0.9, 0.5, 2.7]  # z, y, x
TSL2 = [0.9, 0.5, 2.7]
TSL3 = [0, 1.6, 5.9]
TSL4 = [2, 4.5, 0.1]
TSL5 = [0, 2.7, 2.7]
TSL6 = [0.9, 0.5, 2.7]
TSL7 = [2.2, 0.1, 2.7]
TSL8 = [0.9, 0.5, 2.7]
TSL9 = [0.9, 0.5, 2.7]
TSL10 = [1.7, 3.1, 2.7]
list_tsl = [TSL1, TSL2, TSL3, TSL4, TSL5, TSL6, TSL7, TSL8, TSL9, TSL10]

tsl = TSL1
W1 = wsn_lite.wsn('Exp01')
W2 = wsn_lite.wsn('Exp02')
W3 = wsn_lite.wsn('Exp03')
W4 = wsn_lite.wsn('Exp04')
W5 = wsn_lite.wsn('Exp05')
W6 = wsn_lite.wsn('Exp06')
W7 = wsn_lite.wsn('Exp07')
W8 = wsn_lite.wsn('Exp08')
W9 = wsn_lite.wsn('Exp09')
W10 = wsn_lite.wsn('Exp10')
list_W = [W1, W2, W3, W4, W5, W6, W7, W8, W9, W10]

#W.plotWind()
#W.plotTempAndHumi()
#W.plotMoxConcentration()
for i in [1]:
    W = list_W[i]
    W.plotGasMap(map_type='mean', timeframe=[tc-delta/2, tc+delta/2], tsl=list_tsl[i])
    #W.plotGasMap(map_type='median', timeframe=[tc-delta/2, tc+delta/2], tsl=list_tsl[i])
    #W.plotGasMap(map_type='max', timeframe=[tc-delta/2, tc+delta/2], tsl=list_tsl[i])
    #W.plotGasMap(map_type='var', timeframe=[tc-delta/2, tc+delta/2], tsl=list_tsl[i])
    W.plotGasMap(map_type='bouts-freq', timeframe=[tc-delta/2, tc+delta/2], bouts_hl=half_life,
                 bouts_ampthresh=bouts_amp_thresh, tsl=list_tsl[i])
#plt.show()
error_mean = np.empty(10)
error_median = np.empty(10)
error_max = np.empty(10)
error_var = np.empty(10)
error_bout = np.empty(10)

# Compare variance, bout frequency and mean maximum estimate at different time
list_tc = [15, 30,  60, 90]
for k in range(4):
    tc = list_tc[k]
    for i in range(10):
        W = list_W[i]
        error_mean[i] = W.computeError(map_type='mean', timeframe=[tc-delta/2, tc+delta/2], tsl=list_tsl[i])
        #error_median[i] = W.computeError(map_type='median', timeframe=[tc-delta/2, tc+delta/2], tsl=list_tsl[i])
        #error_max[i] = W.computeError(map_type='max', timeframe=[tc-delta/2, tc+delta/2], tsl=list_tsl[i])
        error_var[i] = W.computeError(map_type='var', timeframe=[tc-delta/2, tc+delta/2], tsl=list_tsl[i])
        error_bout[i] = W.computeError(map_type='bouts-freq', timeframe=[tc-delta/2, tc+delta/2], tsl=list_tsl[i],
                                       bouts_hl=half_life, bouts_ampthresh=bouts_amp_thresh)
    ax_coord = [[0, 0], [0, 1], [1, 0], [1, 1]]
    fig, ax = plt.subplots()
    ax.plot(np.linspace(1, 10, num=10), error_mean, 'o-', label='Mean error')
    #ax.plot(np.linspace(1, 10, num=10), error_median, 'o-', label='Median error')
    #ax.plot(np.linspace(1, 10, num=10), error_max, 'o-', label='Max error')
    ax.plot(np.linspace(1, 10,  num=10), error_var, 'o-', label='Variance error')
    ax.plot(np.linspace(1, 10, num=10), error_bout, 'o-', label='Bout error')
    ax.set_xticks(np.arange(1, 11))
    ax.set_title(f'Comparison of mean, variance and bout frequency gas source localisation error after {tc} minutes')
    ax.set_xlabel('Experiment from 1 to 10')
    ax.set_ylabel('Gas source localisation error [m]')
    ax.legend()

# Compare variance, bout frequency and mean maximum estimate when tuning half-life time and bout amp threshold
list_bout_thresh = [0.13, 0.2, 0.4, 0.6, 0.8]
list_half_life = [0.1, 0.2, 0.3, 0.4, 0.5]
for k in range(5):
    tc = 60
    bouts_amp_thresh = list_bout_thresh[k]  # default value
    half_life = 0.25
    for i in range(10):
        W = list_W[i]
        error_mean[i] = W.computeError(map_type='mean', timeframe=[tc-delta/2, tc+delta/2], tsl=list_tsl[i])
        #error_median[i] = W.computeError(map_type='median', timeframe=[tc-delta/2, tc+delta/2], tsl=list_tsl[i])
        #error_max[i] = W.computeError(map_type='max', timeframe=[tc-delta/2, tc+delta/2], tsl=list_tsl[i])
        error_var[i] = W.computeError(map_type='var', timeframe=[tc-delta/2, tc+delta/2], tsl=list_tsl[i])
        error_bout[i] = W.computeError(map_type='bouts-freq', timeframe=[tc-delta/2, tc+delta/2], tsl=list_tsl[i],
                                       bouts_hl=half_life, bouts_ampthresh=bouts_amp_thresh)
    ax_coord = [[0, 0], [0, 1], [1, 0], [1, 1]]
    fig, ax = plt.subplots()
    ax.plot(np.linspace(1, 10, num=10), error_mean, 'o-', label='Mean error')
    #ax.plot(np.linspace(1, 10, num=10), error_median, 'o-', label='Median error')
    #ax.plot(np.linspace(1, 10, num=10), error_max, 'o-', label='Max error')
    ax.plot(np.linspace(1, 10,  num=10), error_var, 'o-', label='Variance error')
    ax.plot(np.linspace(1, 10, num=10), error_bout, 'o-', label='Bout error')
    ax.set_xticks(np.arange(1, 11))
    ax.set_title(fr'Tuning amp. bout threshold (b_thr = {bouts_amp_thresh}) after {tc} minutes')
    ax.set_xlabel('Experiment from 1 to 10')
    ax.set_ylabel('Gas source localisation error [m]')
    ax.legend()

plt.show()
