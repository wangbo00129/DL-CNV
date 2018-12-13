'''
Created on 20180917

@author: dell
'''
import os, sys
import glob
import random
import shutil
from numpy import max, mean, median, array, mat
from pkg.GenerateFakeData import generateFakeMatBy, MAT_SUFFIX
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
            if pointer>=len(all_mats):
                continue
            shutil.copy(all_mats[pointer], path_part)
            pointer += 1


    
def crossValidate(data_set_parent):    
    # This function reads in several parts of the data_set_parent 
    #     and choose each per round as the test set. 
    from pkg.CNN_for_CNV_simpler_net_cross_validation import CNNClass
    #data_set_parent = 'd:/eclipse-workspace/CNN_CNV/data_stride_40/'
    parts = glob.glob(os.path.join(data_set_parent, 'part_*'))
    #pool = multiprocessing.Pool(processes=3)
    for i in parts:  
        #if not 'part_4' in i: continue
        iname = os.path.basename(i) 
        path_training = parts.copy()
        path_testing = i
        path_training.remove(i)
#             pool.apply_async(__main__, ('d:/eclipse-workspace/CNN_CNV/data_stride_40/', 21, 62, 1e-5, 20000, conv1, conv2, 
#                                  mean, zero, 
#                                  'd:/eclipse-workspace/CNN_CNV/data_stride_40/normalize_by_mean_fill_by_0_stride_40_')) 
        # No normalizing. 
        keep_probability = 0.5
#         pool.apply_async(__main__, (path_training, path_testing, 21, 62, 5, 5, 
#                   1e-5, 10000, True, max, zero, keep_probability,
#                   os.path.join(data_set_parent, 'TwoCNN_over_sampling_'+str(keep_probability)+'_normalize_by_max_fill_by_0_stride_40_'+iname+'_as_test_'), '1'))
        useTwoLayers = False
        kernel1size = 5
        kernel2size = 5
        if useTwoLayers:
            prefix_useTwoLayers = 'two'
        else:
            prefix_useTwoLayers = 'one'
        model_dir = os.path.join(data_set_parent, prefix_useTwoLayers+'_kernel_'+
                                 str(keep_probability)+'_oversamp1_4fold_normalizebymean_fillby0_stride25_'+iname+'_as_test_')
        
        print(model_dir+str(kernel1size)+'_'+str(kernel2size))
#         __main__(path_training, path_testing, 21, 62, 5, 5, 
#                 1e-5, 20000, False, mean, zero, keep_probability,
#                 model_dir, None)

        #path_testing = 'D:/eclipse-workspace/CNN_CNV/data_producing_soft_links_mats_stride_25/correct_label'
        CNNClass(path_training, path_testing, 27, 24, kernel1size, kernel2size, 1e-5, 30000, 
                 useTwoLayers, mean, zero, keep_probability, model_dir, '1', 4)
        
#     pool.close()
#     pool.join()
#     
if __name__ == '__main__':
    #input_dir = 'D:/eclipse-workspace/CNN_CNV/data_stride_40/all' 
    input_dir = 'D:/eclipse-workspace/CNN_CNV/data_bigger_training_added_strong_positive_stride_40_dense_fake_data\mixed'
    input_dir = 'D:/eclipse-workspace/CNN_CNV/data_bigger_40plus40/mixed'
    input_dir = 'D:/eclipse-workspace/CNN_CNV/data_producing_soft_links_mats_stride_40/correct_label'
    input_dir = 'D:/eclipse-workspace/CNN_CNV/data_large_model/mixed' 
    input_dir = 'D:/eclipse-workspace/CNN_CNV/data_ERBB2_stride40/mixed'
	input_dir, cross_num, fold_num = sys.argv[1:4]
	
    generate_fake = False
    for i in range(0,cross_num):
        output_dir = os.path.join(os.path.dirname(input_dir), str(fold_num)+'xCV_data_rand_'+str(i))
        if not os.path.exists(output_dir):
            splitDataSet(input_dir, output_dir, fold_num)
            paths_mats = glob.glob(os.path.join(output_dir, 'part_*' ,'*'+MAT_SUFFIX))
            
            if generate_fake:                 
                [generateFakeMatBy(m) for m in paths_mats]
        
        crossValidate(output_dir)
    #crossValidate()