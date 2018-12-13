'''
Created on 20181026

@author: dell
'''
import pickle
from numpy import shape, mat
import numpy as np
from copy import deepcopy
import os
import glob

MAT_SUFFIX = '.mat'
MAT_FAKE_SUFFIX = 'f.mat'


def generateFakeMatBy(path_mat):
    # Set every non-zero element in the mat to zero each time, generating m*n matrices. 
    fr = open(path_mat, 'rb')
    print(path_mat)
    mat_name_no_suffix = os.path.basename(path_mat).replace(MAT_SUFFIX, '')
    parent_name = os.path.dirname(path_mat)
    mat_in = pickle.load(fr, encoding = 'iso-8859-1')
    fr.close()
    m, n = shape(mat_in)
    for i in range(m):
        for j in range(n):
            if mat_in[i,j]==0: 
                continue
            mat_out = deepcopy(mat_in)
            mat_out[i,j] = 0
            path_mat_out = os.path.join(parent_name, mat_name_no_suffix+'_'+str(i)+'_'+str(j)+MAT_SUFFIX)
            fw = open(path_mat_out, 'wb')
            pickle.dump(mat_out, fw)
            fw.close()
            
            
if __name__ == '__main__':
    path_parent_mats = 'D:/eclipse-workspace/CNN_CNV/data_bigger_training_added_strong_positive_stride_40_dense_fake_data/mixed'
    paths_mats = glob.glob(os.path.join(path_parent_mats, '*'+MAT_SUFFIX))
    [generateFakeMatBy(m) for m in paths_mats]