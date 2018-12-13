'''
Created on 20180917

@author: dell
'''
import os 
import glob
import random
import shutil
import multiprocessing
from numpy import max, mean, median, array, mat
#from pkg.DataSet import mymedian

def mymedian(input_mat):
    return median(input_mat, axis=1)
def zero(num):
    return 0.0
def one(num):
    return 1.0

def splitDataSet(input_dir, output_dir, k_fold=10):
    all_mats = glob.glob(os.path.join(input_dir, '*'))
    part_size = round(float(len(all_mats))/k_fold)
    random.shuffle(all_mats)
    pointer = 0
    for i in range(k_fold):
        path_part = os.path.join(output_dir, 'part_'+str(i))
        os.makedirs(path_part, exist_ok=True)
        for _ in range(part_size):
            shutil.copy(all_mats[pointer], path_part)
            pointer += 1


    
def crossValidate(data_set_parent):    
    # This function reads in several parts of the data_set_parent 
    #     and choose each per round as the test set. 
    from pkg.NN_before_CNN_for_CNV_simpler_net_cross_validation import __main__
    #data_set_parent = 'd:/eclipse-workspace/CNN_CNV/data_stride_40/'
    parts = glob.glob(os.path.join(data_set_parent, 'part*'))
    #pool = multiprocessing.Pool(processes=3)
    for i in parts: 
        iname = os.path.basename(i) 
        path_training = parts.copy()
        path_testing = [i]
        path_training.remove(i)
#             pool.apply_async(__main__, ('d:/eclipse-workspace/CNN_CNV/data_stride_40/', 21, 62, 1e-5, 20000, conv1, conv2, 
#                                  mean, zero, 
#                                  'd:/eclipse-workspace/CNN_CNV/data_stride_40/normalize_by_mean_fill_by_0_stride_40_')) 
        # No normalizing. 
        keep_probability = 0.75
#         pool.apply_async(__main__, (path_training, path_testing, 21, 62, 5, 5, 
#                   1e-5, 10000, True, max, zero, keep_probability,
#                   os.path.join(data_set_parent, 'TwoCNN_over_sampling_'+str(keep_probability)+'_normalize_by_max_fill_by_0_stride_40_'+iname+'_as_test_'), '1'))
        model_dir = os.path.join(data_set_parent, 'NNBefore_over_sampling_'+str(keep_probability)+'_normalize_by_file_size_fill_by_0_stride_40_'+iname+'_as_test_')
        
        print(model_dir)
        __main__(path_training, path_testing, 21, 62, 5, 5, 
                1e-5, 5000, False, 'file_size', zero, keep_probability,
                model_dir, '1')
#     pool.close()
#     pool.join()        
#     
if __name__ == '__main__':
    #input_dir = 'D:/eclipse-workspace/CNN_CNV/data_stride_40/all' 
    for i in range(0,10):
        output_dir = 'D:/eclipse-workspace/CNN_CNV/data_stride_40/5xCV_data_rand_'+str(i)
        #splitDataSet(input_dir, output_dir, 5)
        crossValidate(output_dir)
    #crossValidate()