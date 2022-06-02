import wsn_lite_webots
import matplotlib.pyplot as plt
import numpy as np

X_GRID = 29
Y_GRID = 7
Z_GRID = 1

# Time
delta = 5.0/60  # minutes
tc = 2.5/60  # minutes

# Parameters
half_life = 0.01  # seconds DEFAULT VALUE IS 0.25 [s]
bouts_amp_thresh = 5e3  # DEFAULT VALUE IS 0.13

# Load data
W1 = wsn_lite_webots.wsn('2D_cluttered')
W2 = wsn_lite_webots.wsn('2D_windspeed_0.25')
W3 = wsn_lite_webots.wsn('2D_windspeed_0.5')
W4 = wsn_lite_webots.wsn('2D_windspeed_0.75')
W5 = wsn_lite_webots.wsn('2D_windspeed_1.0')

list_W = [W1, W2, W3, W4, W5]
list_bouts_amp_thresh = [None]*len(list_W)

# True Source Location
TSL = [0.1, 2, 15]  # (z, y, x)

# Plot results
for i in range(len(list_W)):
    list_W[i].plotGasMap(map_type='mean', timeframe=[tc-delta/2, tc+delta/2], tsl=TSL)

    list_bouts_amp_thresh[i] = list_W[i].compute_bouts_amps_threshold(timeframe=[tc - delta / 2, tc + delta / 2],
                                                                      hl=half_life, method='percentage', plot=False)
    list_W[i].plotGasMap(map_type='bouts-freq', timeframe=[tc-delta/2, tc+delta/2], bouts_hl=half_life,
                 bouts_ampthresh=list_bouts_amp_thresh[i], tsl=TSL)

plt.show()
