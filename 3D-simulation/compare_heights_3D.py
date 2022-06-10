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
bout_amp_meth = 'knee'

total_error_mean = []
total_error_bouts = []

# Load data
for h in ['h1', 'h2', 'h3']:
    directory = os.path.join(rf"C:\Users\Malik\Documents\Ecole\EPFL\Master\MA2\Semester project\GSL_using_mox_sensors\3D-simulation\logs_webots\{h}")
    for root, dirs, files in os.walk(directory):
        list_W = []
        list_tsl = []
        for file in files:
            if file.endswith(".csv"):
                list_W.append(wsn_lite_webots.wsn(file, h))
                m1 = re.search('spx', file).span()  # spx coord.
                m2 = re.search('_spy', file).span()  # spy coord.
                m3 = re.search('_spz', file).span()  # spz coord.
                m4 = re.search(f'_{h}', file).span()  # h coord.
                if m1 and m2 and m3 and m4:
                    list_tsl.append([float(file[m3[1]:m4[0]]), float(file[m2[1]:m3[0]]), float(file[m1[1]:m2[0]])])
                else:
                    quit('WARNING: problem when reading files')

    # Initialize lists
    list_bouts_amp_thresh = []
    error_bout = []
    error_mean = []

    # Compute error of mean and bouts
    i = 0
    for W in list_W:
        list_bouts_amp_thresh.append(W.compute_bouts_amps_threshold(timeframe=[tc - delta / 2, tc + delta / 2],
                                                                    hl=half_life, method=bout_amp_meth, plot=False))

        error_mean.append(W.computeError(map_type='mean', timeframe=[tc-delta / 2, tc+delta / 2], tsl=list_tsl[i]))
        error_bout.append(W.computeError(map_type='bouts-freq', timeframe=[tc-delta/2, tc+delta/2], tsl=list_tsl[i],
                                         bouts_hl=half_life, bouts_ampthresh=list_bouts_amp_thresh[i]))
        i = i+1

    # Plot results
    if h == 'h2':
        for exp_numb in range(len(list_W)):
            list_W[exp_numb].plotGasMap(map_type='mean', timeframe=[tc-delta/2, tc+delta/2], tsl=list_tsl[exp_numb])
            list_W[exp_numb].plotGasMap(map_type='bouts-freq', timeframe=[tc-delta/2, tc+delta/2], bouts_hl=half_life,
                                        bouts_ampthresh=list_bouts_amp_thresh[exp_numb], tsl=list_tsl[exp_numb])

    # Store error for each wind speed
    total_error_mean.append(error_mean)
    total_error_bouts.append(error_bout)

    print(f'Height {h} done !')

# Plot box plots of mean and bouts
error_max = np.amax(np.array([total_error_mean, total_error_bouts]))

plt.figure()
plt.boxplot(total_error_mean)
plt.title('Localization error of mean concentration for various heights in 3D')
plt.xlabel('Height [m]')
plt.ylabel('Localization error [m]')
plt.xticks([1, 2, 3], ['h1 = 0.9', 'h2 = 1.25', 'h3 = 1.41'])
#plt.ylim(0, error_max+0.1*error_max)
plt.savefig('boxplot_mean_3D.png')

plt.figure()
plt.boxplot(total_error_bouts)
plt.title('Localization error of bouts algorithm for various heights in 3D')
plt.xlabel('Height [m]')
plt.ylabel('Localization error [m]')
plt.xticks([1, 2, 3], ['h1 = 0.9', 'h2 = 1.25', 'h3 = 1.41'])
# plt.ylim(0, error_max+0.1*error_max)
plt.savefig('boxplot_bouts_3D.png')


plt.show()
