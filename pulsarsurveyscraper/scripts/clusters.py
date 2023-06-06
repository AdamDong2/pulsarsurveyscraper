#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os.path
import os
from datetime import timedelta as td
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
import datetime
from datetime import datetime as dt
import gc
import multiprocessing as mp
import csv
import frb_L2_L3.actors.dm_checker as dmcheck
import re
import pytz
from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist

def assign(mat,err,i,j):
    if err[i] > err[j]:
        mat[i,j] = err[i]
    else:
        mat[i,j] = err[j]
    return mat

class clusters:
    #this class is for a single cluster
    def __init__(self,ra,ra_error,dec,dec_error,dm,dm_error,c_num):
        self.localised_pos_ra_deg = np.array(ra)
        self.localised_pos_dec_deg = np.array(dec)
        self.ra_error = np.array(ra_error)
        self.dec_error = np.array(dec_error)
        self.dm = np.array(dm)
        self.dm_error = np.array(dm_error)
        self.c_num = c_num

    def start_dbscan(self):
        ra_feed = np.vstack((self.localised_pos_ra_deg,self.localised_pos_ra_deg)).T
        dec_feed = np.vstack((self.localised_pos_dec_deg,self.localised_pos_dec_deg)).T
        dm_feed = np.vstack((self.dm,self.dm)).T

        ra_mat = np.divide(distance_matrix(ra_feed,ra_feed),np.sqrt(2))
        dec_mat = np.divide(distance_matrix(dec_feed,dec_feed),np.sqrt(2))
        dm_mat = np.divide(distance_matrix(dm_feed,dm_feed),np.sqrt(2))

        #need to fill an error matrix out
        ra_sig = np.empty_like(ra_mat)
        dec_sig = np.empty_like(dec_mat)
        dm_sig = np.empty_like(dm_mat)

        #the only way is to look through
        for i in range(len(self.dm)):
            for j in range(len(self.dm)):
                ra_sig = assign(ra_sig,self.ra_error,i,j)
                dec_sig = assign(dec_sig,self.dec_error,i,j)
                dm_sig = assign(dm_sig,self.dm_error,i,j)

        #this is the precomputed distance matrix
        d_mat = np.maximum(np.divide(ra_mat,ra_sig),np.divide(dec_mat,dec_sig))
        d_mat = np.maximum(d_mat,np.divide(dm_mat,dm_sig))

        clustering = DBSCAN(eps=2,min_samples=2,metric="precomputed").fit(d_mat)
        self.labels = clustering.labels_
        print('fine tune finished for cluster ',self.c_num)
        # unique = set(self.labels)
        # for i in unique:
        #     mask = i==self.labels
        #     plt.scatter(self.localised_pos_ra_deg[mask],self.localised_pos_dec_deg[mask])
        # plt.show()

    def plot_and_save(self,directory="fine_tuning/"):
        unique = set(self.labels)
        plt.figure()
        for i in unique:
            mask = i==self.labels
            plt.scatter(self.localised_pos_ra_deg[mask],self.localised_pos_dec_deg[mask])
        plt.xlabel('RA')
        plt.ylabel('Dec')
        plt.savefig(directory+'/'+str(self.c_num)+'_fine_tune.png')
        plt.close()
