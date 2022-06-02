#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 12:56:58 2018

@author: root
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PLOT_SIGNAL = True

def compute_bouts(sd, sd_thr=0.0):
    """
    Internal method to compute the "bouts" in the EWMA-filtered time-derivative (sd) of the smoothed signal. 
    The argument sd_thr is the threshold that determines the positive part of the derivative.
    """
    if np.array(sd).all() == 0:
        posneg = np.array([[1], [2]])

    else:
        sd = np.array(sd)[0]
        sd_pos = sd >= sd_thr  # positive part of the derivative
        signchange = np.diff(np.array(sd_pos, dtype=int))  #Change neg->pos=1, pos->neg=-1.
        #print(f"signchange = {signchange}")
        pos_changes = np.nonzero(signchange > 0)[0]
        #print(f"pos_changes = {pos_changes}")
        neg_changes = np.nonzero(signchange < 0)[0]
        # have to ensure that first change is positive, and every pos. change is complemented by a neg. change
        if pos_changes[0] > neg_changes[0]: #first change is negative
            #discard first negative change
            neg_changes = neg_changes[1:]
        if len(pos_changes) > len(neg_changes): # lengths must be equal
            difference = len(pos_changes) - len(neg_changes)
            pos_changes = pos_changes[:-difference]
        posneg = np.zeros((2, len(pos_changes)))
        posneg[0, :] = pos_changes
        posneg[1, :] = neg_changes
    #print(posneg)
    return posneg


def compute_bouts_RT(raw_signal, fs=10, hl=0.3, ampthresh=0.0, sd_thr=0.0): 
    """
    Real-time bout computation.
    The Gaussian smoothing filter of Schmuker's algorithm is replaced by an EWMA filter.
    The argument "raw_signal" is the unsmoothed sensor response, "fs" is the sampling frequency (Hz), 
    "hl" is the half-life time (s) and "sd_thr" is the threshold to determine when the derivative is positive.
    """
    if PLOT_SIGNAL:
        fig, axes = plt.subplots(2, 2)
        axes[0][0].plot(raw_signal)
        axes[0, 0].set_title('Raw signal')

    raw_signal = pd.DataFrame(raw_signal)  # convert raw_signal from np.ndarray to pd.DataFrame
    print(raw_signal)
    s = pd.DataFrame.ewm(raw_signal, halflife=hl*fs, adjust=False, ignore_na=True).mean()
    s = np.array(s).T  # convert s to np.array
    if PLOT_SIGNAL:
        axes[0][1].plot(s[0])
        axes[0, 1].set_title('Signal smoothed')

    sd = fs * np.diff(s)
    if PLOT_SIGNAL:
        axes[1][0].plot(sd[0])
        axes[1, 0].set_title('Derivated Signal')

    hl = 0.3
    sd = pd.DataFrame(sd)  # convert sd from np.ndarray to pd.DataFrame
    print(sd)
    sds = pd.DataFrame.ewm(sd, halflife=hl*fs, adjust=False, ignore_na=True, axis=0).mean()
    print(sds)
    if PLOT_SIGNAL:
        axes[1][1].plot(np.array(sds)[0])
        axes[1, 1].set_title('Smoothed derivative signal')

    bouts = compute_bouts(sds, sd_thr).astype(int)
    amps = np.zeros(bouts.shape)
    bouts = bouts.tolist()
    sds = np.array(sds).tolist()[0]
    # Calculate amplitude of bouts
    amps[0] = [sds[int(e)] for e in bouts[0]]
    amps[1] = [sds[int(e)] for e in bouts[1]]
    amps = np.diff(amps, axis=0)[0]
    # amps = np.hstack(np.diff(amps).flat) why do np.hstack and .flat ?

    bouts = np.array(bouts).reshape(1, -1)
    bouts_filt = bouts[amps>ampthresh, :]
    amps_filt = amps[amps>ampthresh]
    print(f'bouts filtered\n{bouts_filt}\namps filtered\n{amps_filt}')
    if PLOT_SIGNAL:
        fig.tight_layout()
        plt.show()

    return bouts_filt, amps_filt
