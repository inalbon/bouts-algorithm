# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 11:54:39 2017

@author: jburgues
"""
from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import csv
import log_wsn_webots
from bouts import compute_bouts_RT
from kneed import KneeLocator

X_GRID = 29
Y_GRID = 7
Z_GRID = 1

class wsn:

    def __init__(self, log_name):
        L = log_wsn_webots.LogWsn(log_name)
        L.parseLog()
        
        self.t_s = L.t_s
        self.c_ppm = L.c_ppm
        self.coords = L.coords
        self.nr_of_samples, self.nr_of_sensors = L.c_ppm.shape
        self.fs = 1/np.mean(np.diff(self.t_s))
        self.T = self.t_s[[0,-1]]
      
    # def _getTimeIndex(self, T, t_tol=None):  
    # Extract temporal indices corresponding to a time frame.
    # T: 1x2 vector specifying the start and end times (minutes) of the time 
    # frame (e.g. T=[5, 10]). 
    def _getTimeIndex(self, timeframe=None): 
        """
        Internal method to obtain the sample index from a time vector.
        """         
        t_min = self.t_s / 60.0
        
        if timeframe is None:
            timeframe = t_min[[0, -1]].flatten()
            
        t_idx = (t_min > timeframe[0]) & (t_min < timeframe[1])
            
        return t_idx.flatten()
    
    def _smooth_signal(signal, winsize, plot=False):
        """
        Internal method to smooth a signal using a Gaussian filter
        """
        
        df = pd.DataFrame(signal)
        signal_smooth = df.rolling(winsize, win_type='gaussian', min_periods=1, center=True)
        signal_smooth = signal_smooth.mean(std=winsize).values[:, 0]
        
        return signal_smooth

    
    def plotMoxConcentration(self, ax=None, sensor='all', timeframe=None, smooth=False):
        if not ax or ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            fig = plt.gcf()
            
        tidx = self._getTimeIndex(timeframe)
        c_ppm = self.c_ppm[tidx,:]
        t_s = self.t_s[tidx]
            
        if smooth:  # smooth signals
            winsize_s = 20
            winsize = int(round(winsize_s * self.fs))
            c_ppm_smooth = np.zeros_like(c_ppm)            
            for i in range(self.nr_of_sensors):
                c_ppm_smooth[:,i] = self._smooth_signal(c_ppm[:,i], winsize)
                c_ppm = c_ppm_smooth
        
        plt1 = ax.plot(t_s/60, c_ppm)
        ax.set_ylabel('Instantenous concentration (ppm)')
        ax.set_xlabel('Time (min)')
        plt.xlim([-1, self.t_s[-1]/60 + 1])
 
        return fig, ax, plt1

    def _computeGasMap(self, mapType, timeframe, hl=0.25, bout_ampl=0):   
        t_idx = self._getTimeIndex(timeframe)
        t_s = self.t_s[t_idx]
        c_ppm = self.c_ppm[t_idx, :]
            
        # Build grid of mean concentration
        nx = X_GRID
        ny = Y_GRID
        nz = Z_GRID
        gridmap = np.zeros([nz, ny, nx], dtype=float)

        for i in range(self.nr_of_sensors):
            if mapType == 'mean':
                val = np.mean(c_ppm[:,i])
            elif mapType == 'median':
                val = np.median(c_ppm[:,i])
            elif mapType == 'max':
                val = np.max(c_ppm[:,i])
            elif mapType == 'var':
                val = np.var(c_ppm[:,i])
            elif mapType.startswith('bouts'):
                # bout count    
                filtered_bouts, amps = compute_bouts_RT(c_ppm[:, i], i, self.fs, hl=hl, ampthresh=bout_ampl)
                pbc = filtered_bouts.shape[0]
                if mapType.endswith('freq'):
                    t_diff = (t_s[-1]-t_s[0])#/60  # seconds
                    val = pbc/t_diff  # bout frequency (bouts/seconds)
                elif mapType.endswith('ampl'):
                    val = np.mean(amps)
            
            x,y,z = self.coords[i,:]
            #gridmap[z,ny-y-1,x] = val
            gridmap[z, y, x] = val

        # Export data to display in MatLab
        #temp = gridmap
        #temp = np.array(temp).reshape(1, -1)
        #print(temp)
        #np.savetxt('export.csv', temp, '%.10f', delimiter=',')

        return gridmap
    
    
    def _plotGasMapSlice(self, fig, ax, gmap_slice, xlbl = True, ylbl = True, 
                         interp='spline36', cm='YlOrRd', cb_lbl='ppm',
                         norm=None, clims=None):
        if clims is None:
            clims = [0, np.max(gmap_slice)]
        if norm is None:
            norm=mpl.colors.Normalize(clims[0], clims[1], True)

        extent = [1, 15, 0.5, 3.5]
        im_map = ax.imshow(gmap_slice, extent=extent, aspect='equal',
                        norm=norm, origin='lower', interpolation=interp,  cmap=cm)

        #ax.invert_xaxis()
        #ax.invert_yaxis()
        ax.tick_params(labelsize=5)
        #ax.xaxis.tick_top()
        plt.setp(ax, xticks=[0, 4, 8, 12, 16], yticks=[0, 2, 4])

        if ylbl:
            ax.set_ylabel('Y (m)', fontsize=7)
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
        if xlbl:
            ax.set_xlabel('X(m)', fontsize=7)
        else:
            plt.setp(ax.get_xticklabels(), visible=False)
        #ax_z.xaxis.set_label_position('top')

        # color bar 
        if cb_lbl is not None:
            #cax = fig.add_axes([0.95, 0.1, 0.03, 0.79])
            cax = plt.axes([0.7, 0.2, 0.03, 0.7])  # [left, bottom, width, height]
            cb = fig.colorbar(im_map, cax=cax) 
            cb.set_label(cb_lbl)
            plt.subplots_adjust(bottom=0.2, right=0.4, top=0.9, hspace=0.7)
            
        return im_map

    def plotGasMap(self, ax=None, map_type='mean', timeframe=None, tsl=None, bouts_hl=0.25,
                   bouts_ampthresh=0.13, interp='spline36', cm='YlOrRd',
                   norm=None, clims=None):
        """
        Plots one of the following maps: mean, median, max, var, bouts-freq or bouts-ampl
        in the timeframe specified. The bouts parameters are "bouts_hl" (half-life time (s))
        and "bouts_ampthresh" (amplitude threshold (ppm/s)).
        """
        gmap = self._computeGasMap(map_type, timeframe, bouts_hl, bouts_ampthresh)

        cb_lbl = 'None'
        if map_type == 'mean':
            cb_lbl = 'Mean response (ppm)'
        elif map_type == 'median':
            cb_lbl = 'Median response (ppm)'
        elif map_type == 'max':
            cb_lbl = 'Max response (ppm)'
        elif map_type == 'var':
            cb_lbl = 'Variance of response (ppm^2)'
        elif map_type == 'bouts-freq':
            cb_lbl = 'Bout frequency (bouts/sec)'
        elif map_type == 'bouts-ampl':
            cb_lbl = 'Mean bout amplitude (ppm/s)'

        if not ax or ax is None:
            fig, ax = plt.subplots(figsize=(4.0, 10.0), dpi=150)
        else:
            fig = plt.gcf()

        if clims is None:
            clims = [0, np.max(gmap)]

        z_real = [0.1, 1.2, 2.23]

        for z in range(1):
            im = self._plotGasMapSlice(fig, ax, gmap[0, :, :],
                                       clims=clims, cm=cm, norm=norm, cb_lbl=cb_lbl)
            ax.text(20, 2.0, 'Z = {0:.2f} m'.format(z_real[z]), rotation=0)

        # display true source location (tsl)
        if tsl is not None:
            ax.plot(tsl[2], tsl[1], marker='.', markerfacecolor="tab:green",
                                       markeredgecolor="tab:green", markersize=2.5)

        #print(f'TSL = ({tsl[2]}, {tsl[1]})')

        # display maximum estimate
        index = np.unravel_index(np.argmax(np.array(gmap), axis=None), np.array(gmap).shape)
        #print(f'max value = ({index[2]}, {index[1]}, {index[0]})')
        ax.plot(index[2]/2+1, index[1]/2+0.5, marker='.',
                                  markerfacecolor="tab:blue", markeredgecolor="tab:blue", markersize=2.5)
        #ax[inv_ax[index[0]]].scatter(index[2]/2+1, index[1]/2+0.5, color='r')
        ax.axis(xmin=0, xmax=16, ymin=0, ymax=4)
        return ax, im


    def computeError(self, map_type, timeframe, tsl, bouts_hl=None, bouts_ampthresh=None):
        gmap = self._computeGasMap(map_type, timeframe, bouts_hl, bouts_ampthresh)
        index = np.unravel_index(np.argmax(np.array(gmap), axis=None), np.array(gmap).shape)
        sensor = [1, 3, 5]
        z_real = [0.45, 1.2, 2.23]
        est_source = np.array([z_real[index[0]], sensor[index[1]], sensor[index[2]]])  # (z, y, x)
        print(f'estimated source {est_source} and tsl {tsl}')
        error = np.linalg.norm(est_source-np.array(tsl))
        return error

    def compute_bouts_amps_threshold(self, timeframe, hl=0.25, method=None, plot=None):
        t_idx = self._getTimeIndex(timeframe)
        t_s = self.t_s[t_idx]
        c_ppm = self.c_ppm[t_idx, :]
        total_amps = []
        # if plot:
        #     plt.figure()
        #     plt.xlabel('bouts detected')
        #     plt.ylabel('bouts amplitude')
        for i in range(self.nr_of_sensors):
            # bout count
            filtered_bouts, amps = compute_bouts_RT(c_ppm[:, i], i, self.fs, hl=hl, ampthresh=0.0)
            # if plot:
            #     plt.plot(amps, label=f'sensor {i}')
            #     plt.legend()
            total_amps.append(amps.tolist())

        total_amps = [item for sublist in total_amps for item in sublist]  # flatten the list
        if plot:
            plt.figure()
            plt.plot(total_amps)
            plt.xlabel('bouts detected by all sensors')
            plt.ylabel('bouts amplitude')

        # Compute amps threshold
        if method == 'mean':
            threshold = np.mean(total_amps)
        if method == 'median':
            threshold = np.median(total_amps)
        if method == 'middle':
            threshold = (max(total_amps)-min(total_amps))/2
        if method == 'percentage':
            total_amps_sorted = sorted(total_amps)
            if plot:
                plt.figure()
                plt.plot(total_amps_sorted)
            threshold = total_amps_sorted[int(len(total_amps)*0.9)]
            print(threshold)
        if method == 'knee':
            total_amps_sorted = sorted(total_amps)
            kl = KneeLocator(range(len(total_amps_sorted)), total_amps_sorted, curve='convex')
            if plot:
                kl.plot_knee()
            threshold = total_amps_sorted[kl.knee]
            print(threshold)
        return threshold


