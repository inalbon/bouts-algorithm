import wsn_lite
import matplotlib.pyplot as plt
import numpy as np

# Time
delta = 5.0  # minutes
tc = 60  # minutes

# Parameters
half_life = 0.3  # seconds DEFAULT VALUE IS 0.25 [s]
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

plt.plot(W1.t_s[0:100], W1.c_ppm[0:100])
plt.xlabel('time [s]')
plt.ylabel('concentration [ppm]')
plt.title('Concentration of 9 sensors')

W7.plotGasMap(map_type='bouts-freq', timeframe=[tc - delta / 2, tc + delta / 2], bouts_hl=half_life,
             bouts_ampthresh=bouts_amp_thresh, tsl=list_tsl[6])
W7.plotGasMap(map_type='mean', timeframe=[tc - delta / 2, tc + delta / 2], tsl=list_tsl[6])

plt.show()