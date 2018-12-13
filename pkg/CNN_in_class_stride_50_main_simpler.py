import os
from pkg.CNN_for_CNV_simpler_net import __main__
import multiprocessing
from numpy import max, mean, median

def zero(num):
    return 0.0
def one(num):
    return 1.0

if __name__=='__main__':
    data_set_parent = 'd:/eclipse-workspace/CNN_CNV/data_stride_40/'
    pool = multiprocessing.Pool(processes=1)
    for conv1 in range(5,7):
        for conv2 in range(1,2):
#             pool.apply_async(__main__, ('d:/eclipse-workspace/CNN_CNV/data_stride_40/', 21, 62, 1e-5, 20000, conv1, conv2, 
#                                  mean, zero, 
#                                  'd:/eclipse-workspace/CNN_CNV/data_stride_40/normalize_by_mean_fill_by_0_stride_40_')) 
            # No normalizing. 
            pool.apply_async(__main__, (data_set_parent, 21, 62, 1e-5, 20000, conv1, conv2, 
                     mean, zero, 
                     os.path.join(data_set_parent, 'simpler_normalize_by_mean_fill_by_0_stride_40_')))
    pool.close()
    pool.join()