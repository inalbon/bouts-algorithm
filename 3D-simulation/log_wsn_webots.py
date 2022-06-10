#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 11:09:06 2018

@author: root
"""

import csv
import numpy as np
import re
from os.path import join, dirname, realpath

X_GRID = 16
Y_GRID = 5
Z_GRID = 3


class LogWsn(object):
    def __init__(self, log_name, ws):
        self.dir_path = dirname(realpath(__file__))
        print(self.dir_path)
        self.log_name = log_name
        self.t_s = None
        self.c_ppm = None
        self.coords = None
        self.ws = ws

    def parseLog(self):
        exp_folder = join(self.dir_path, f'logs_webots/ws_{self.ws}', self.log_name)

        # Read the log and save it into a list    
        with open(exp_folder, 'r', encoding='mac_roman') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            data_with_header = list(csvreader)
        
        # Parse the header to find the coordinates of each sensor
        header = data_with_header[0]
        coords = []
        for i in range(len(header)):
            m = re.match('C_', header[i])
            if m:
                if (i < 51) or (i > 80 and i < 131) or (i > 160 and i < 211):
                    coords.append([int(header[i][2]), int(header[i][3]), int(header[i][4])])
                else:
                    coords.append([int(header[i][2:4]), int(header[i][4]), int(header[i][5])])

        self.coords = np.array(coords)
        nr_of_sensors = len(coords)

        # Parse the rest of the data
        data = data_with_header[1:]
        nr_of_samples = len(data)
        
        self.t_s = np.zeros(nr_of_samples)  
        self.c_ppm = np.zeros((nr_of_samples, nr_of_sensors))
        for i in range(nr_of_samples):
            self.t_s[i] = data[i][0]
            self.c_ppm[i, :] = data[i][1:(nr_of_sensors+1)]  # Concentration (ppm)
