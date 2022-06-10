#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 12:56:58 2018

@author: root
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PLOT_SIGNAL = False
PLOT_FILTER = False
X_GRID = 29
Y_GRID = 7
Z_GRID = 1

def compute_bouts(sd, sd_thr=0.0):
    """
    Internal method to compute the "bouts" in the EWMA-filtered time-derivative (sd) of the smoothed signal. 
    The argument sd_thr is the threshold that determines the positive part of the derivative.
    """
    if np.array(sd).all() == 0:
        posneg = np.array([[1], [2]])

    else:
        sd = np.array(sd).reshape(1, -1)[0]
        sd_pos = sd >= sd_thr  # positive part of the derivative
        signchange = np.diff(np.array(sd_pos, dtype=int))  #Change neg->pos=1, pos->neg=-1 ?
        pos_changes = np.nonzero(signchange > 0)[0]
        neg_changes = np.nonzero(signchange < 0)[0]
        if pos_changes.size == 0:
            posneg = np.array([[1], [2]])
            return posneg

        # have to ensure that first change is positive, and every pos. change is complemented by a neg. change
        if pos_changes[0] > neg_changes[0]:  #first change is negative
            #discard first negative change
            neg_changes = neg_changes[1:]
        if len(pos_changes) > len(neg_changes):  # lengths must be equal
            difference = len(pos_changes) - len(neg_changes)
            pos_changes = pos_changes[:-difference]
        posneg = np.zeros((2, len(pos_changes)))
        posneg[0, :] = pos_changes
        posneg[1, :] = neg_changes
    return posneg


def compute_bouts_RT(raw_signal, sensor_nb, fs=10, hl=0.25, ampthresh=0.0, sd_thr=0.0):
    """
    Real-time bout computation.
    The Gaussian smoothing filter of Schmuker's algorithm is replaced by an EWMA filter.
    The argument "raw_signal" is the unsmoothed sensor response, "fs" is the sampling frequency (Hz), 
    "hl" is the half-life time (s) and "sd_thr" is the threshold to determine when the derivative is positive.
    """
    if PLOT_FILTER:
        plt.figure()
        plt.plot(raw_signal)
        plt.title('Raw signal')
        plt.xlabel('Iterations')
        plt.ylabel('Concentration')

    if PLOT_SIGNAL:
        fig, axes = plt.subplots(2, 2)
        axes[0][0].plot(raw_signal)
        axes[0, 0].set_title(f'Raw signal ({sensor_nb//Y_GRID}, {sensor_nb%Y_GRID}, 0)')

    raw_signal = pd.DataFrame(raw_signal)  # convert raw_signal from np.ndarray to pd.DataFrame
    s = pd.DataFrame.ewm(raw_signal, halflife=hl*fs, adjust=False, ignore_na=True).mean()
    s = np.array(s).reshape(1, -1)[0]  # convert s to np.array

    if PLOT_FILTER:
        plt.figure()
        plt.plot(s)
        plt.title(r'Smoothed signal ($\tau_{half}$ = 0.1 [s])')
        plt.xlabel('Iterations')
        plt.ylabel('Concentration')

    if PLOT_SIGNAL:
        axes[0][1].plot(s)
        axes[0, 1].set_title(f'Smoothed signal ({sensor_nb//Y_GRID}, {sensor_nb%Y_GRID}, 0)')

    sd = fs * np.diff(s)

    if PLOT_SIGNAL:
        axes[1][0].plot(sd)
        axes[1, 0].set_title(f'Derivative signal ({sensor_nb//Y_GRID}, {sensor_nb%Y_GRID}, 0)')

    sd = pd.DataFrame(sd)  # convert sd from np.ndarray to pd.DataFrame
    sds = pd.DataFrame.ewm(sd, halflife=hl*fs, adjust=False, ignore_na=True).mean()

    if PLOT_SIGNAL:
        axes[1][1].plot(np.array(sds))
        axes[1, 1].set_title(f'Smoothed derivative signal ({sensor_nb//Y_GRID}, {sensor_nb%Y_GRID}, 0)')

    bouts = compute_bouts(sds, sd_thr).astype(int)

    if PLOT_SIGNAL:
        pos_changes = bouts[0, :]
        neg_changes = bouts[1, :]
        print(pos_changes, neg_changes)
        for b in range(len(pos_changes)):
            axes[1, 1].plot(np.arange(pos_changes[b], neg_changes[b] + 1), np.array(sds)[pos_changes[b]:neg_changes[b] + 1],
                     color='tab:red')
            axes[1, 1].plot(pos_changes[b], np.array(sds)[pos_changes[b]], marker='.', markerfacecolor='None',
                            markeredgecolor="tab:green", markersize=7)
            axes[1, 1].plot(neg_changes[b], np.array(sds)[neg_changes[b]], marker='.', markerfacecolor='None',
                            markeredgecolor="tab:red", markersize=7)
            axes[0, 1].plot(np.arange(pos_changes[b], neg_changes[b] + 1),
                            np.array(s)[pos_changes[b]:neg_changes[b] + 1],
                            color='tab:red')
            axes[0, 1].plot(pos_changes[b], np.array(s)[pos_changes[b]], marker='.', markerfacecolor='None',
                            markeredgecolor="tab:green", markersize=7)
            axes[0, 1].plot(neg_changes[b], np.array(s)[neg_changes[b]], marker='.', markerfacecolor='None',
                            markeredgecolor="tab:red", markersize=7)

    amps = np.zeros(bouts.shape)
    bouts = bouts.tolist()
    sds = np.array(sds).reshape(1, -1).tolist()[0]
    # Calculate amplitude of bouts
    amps[0] = [sds[int(e)] for e in bouts[0]]
    amps[1] = [sds[int(e)] for e in bouts[1]]
    amps = np.diff(amps, axis=0)[0]
    #print('amps non filtered', amps)
    # amps = np.hstack(np.diff(amps).flat) why do np.hstack and .flat ?

    bouts = np.array(bouts).T
    bouts_filt = bouts[amps>ampthresh, :]
    amps_filt = amps[amps>ampthresh]
    #print(f'bouts filtered\n{bouts_filt}\namps filtered\n{amps_filt}')
    if PLOT_SIGNAL:
        fig.tight_layout()
        plt.show()

    return bouts_filt, amps_filt
