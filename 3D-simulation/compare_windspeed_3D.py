import wsn_lite_webots
import matplotlib.pyplot as plt
import numpy as np
import os
import re

X_GRID = 29
Y_GRID = 7
Z_GRID = 1

# Time
delta = 5.0/60  # minutes
tc = 2.5/60  # minutes

# Parameters
half_life = 0.03  # seconds DEFAULT VALUE IS 0.25 [s]
bouts_amp_thresh = 0  # DEFAULT VALUE IS 0.13

total_error_mean = []
total_error_bouts = []

# Load data
for ws in [0.75]:
    directory = os.path.join(rf"C:\Users\Malik\Documents\Ecole\EPFL\Master\MA2\Semester project\GSL_using_mox_sensors\3D-simulation\logs_webots\ws_{ws}")
    for root, dirs, files in os.walk(directory):
        list_W = []
        list_tsl = []
        for file in files:
            if file.endswith(".csv"):
                list_W.append(wsn_lite_webots.wsn(file, ws))
                m1 = re.search('spx', file).span()  # spx coord.
                m2 = re.search('_spy', file).span()  # spy coord.
                m3 = re.search('_spz', file).span()  # spz coord.
                m4 = re.search('_ws_', file).span()  # ws coord.
                if m1 and m2 and m3 and m4:
                    list_tsl.append([float(file[m3[1]:m4[0]]), float(file[m2[1]:m3[0]]), float(file[m1[1]:m2[0]])])
                else:
                    quit('WARNING: problem when reading files')
    print(list_tsl)
    list_bouts_amp_thresh = []
    error_bout = []
    error_mean = []

    # Compute error of mean and bouts
    i = 0
    for W in list_W:
        list_bouts_amp_thresh.append(W.compute_bouts_amps_threshold(timeframe=[tc - delta / 2, tc + delta / 2],
                                                                    hl=half_life, method='knee', plot=False))

        error_mean.append(W.computeError(map_type='mean', timeframe=[tc-delta / 2, tc+delta / 2], tsl=list_tsl[i]))
        error_bout.append(W.computeError(map_type='bouts-freq', timeframe=[tc-delta/2, tc+delta/2], tsl=list_tsl[i],
                                         bouts_hl=half_life, bouts_ampthresh=list_bouts_amp_thresh[i]))
        i = i+1

    # Plot results
    for exp_numb in range(len(list_W)):
        list_W[exp_numb].plotGasMap(map_type='mean', timeframe=[tc-delta/2, tc+delta/2], tsl=list_tsl[exp_numb])
        list_W[exp_numb].plotGasMap(map_type='bouts-freq', timeframe=[tc-delta/2, tc+delta/2], bouts_hl=half_life,
                                    bouts_ampthresh=list_bouts_amp_thresh[exp_numb], tsl=list_tsl[exp_numb])

    # Store error for each wind speed
    total_error_mean.append(error_mean)
    total_error_bouts.append(error_bout)

    print(f'Wind speed of {ws} [m/s] done !')
    print(total_error_mean)
    print(total_error_bouts)

# Plot box plots of mean and bouts
error_max = np.amax(np.array([total_error_mean, total_error_bouts]))

plt.figure()
plt.boxplot(total_error_mean)
plt.title('Performance of mean response for various wind speed')
plt.xlabel('Wind speed [m/s]')
plt.ylabel('Localization error [m]')
plt.xticks([1, 2, 3, 4], [0.25, 0.5, 0.75, 1.0])
#plt.ylim(0, error_max+0.1*error_max)
plt.savefig('boxplot_mean.eps')

plt.figure()
plt.boxplot(total_error_bouts)
plt.title('Performance of bouts algorithm for various wind speed')
plt.xlabel('Wind speed [m/s]')
plt.ylabel('Localization error [m]')
plt.xticks([1, 2, 3, 4], [0.25, 0.5, 0.75, 1.0])
# plt.ylim(0, error_max+0.1*error_max)
plt.savefig('boxplot_bouts.eps')


plt.show()
