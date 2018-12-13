from pkg.CNN_for_CNV import __main__
import multiprocessing
from numpy import max, mean, median

def zero(num):
    return 0.0
def one(num):
    return 1.0

if __name__=='__main__':
    pool = multiprocessing.Pool(processes=4)
    for conv1 in range(5,7):
        for conv2 in range(5,7):
#             pool.apply_async(__main__, ('d:/eclipse-workspace/CNN_CNV/data_stride_40/', 21, 62, 1e-5, 20000, conv1, conv2, 
#                                  mean, zero, 
#                                  'd:/eclipse-workspace/CNN_CNV/data_stride_40/normalize_by_mean_fill_by_0_stride_40_')) 
            # No normalizing. 
            pool.apply_async(__main__, ('d:/eclipse-workspace/CNN_CNV/data_stride_50/', 21, 50, 1e-5, 20000, conv1, conv2, 
                     mean, zero, 
                     'd:/eclipse-workspace/CNN_CNV/data_stride_50/normalize_by_mean_fill_by_0_stride_50_'))
    pool.close()
    pool.join()