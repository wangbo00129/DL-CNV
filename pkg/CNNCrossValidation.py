'''
Created on 20180917

@author: dell
'''
import os, sys
import glob
import random
import shutil
import pickle
from numpy import max, mean, median, array, mat
from pkg.GenerateFakeData import generateFakeMatBy, MAT_SUFFIX
from pkg.CNN_for_CNV_simpler_net_cross_validation import CNNClass
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


    
def crossValidate(data_set_parent, learning_rate, iter_num, keep_probability, kernel_size, over_sampling_class=None, over_sampling_fold=1):    
    # This function reads in several parts of the data_set_parent 
    #     and choose each per round as the test set. 
    #data_set_parent = 'd:/eclipse-workspace/CNN_CNV/data_stride_40/'
    parts = glob.glob(os.path.join(data_set_parent, 'part_*'))
    #pool = multiprocessing.Pool(processes=3)
    for i in parts:  
        #if not 'part_4' in i: continue
        iname = os.path.basename(i) 
        path_training = parts.copy()
        path_testing = i
        path_training.remove(i)
        sample_path_to_get_shape = path_training[0][0]
        try:
            mt = pickle.load(open(sample_path_to_get_shape, 'rb'), encoding='iso-8859-1')
        except:
            mt = pickle.load(open(sample_path_to_get_shape, 'rb'))
        m, n = mt.shape
#             pool.apply_async(__main__, ('d:/eclipse-workspace/CNN_CNV/data_stride_40/', 21, 62, 1e-5, 20000, conv1, conv2, 
#                                  mean, zero, 
#                                  'd:/eclipse-workspace/CNN_CNV/data_stride_40/normalize_by_mean_fill_by_0_stride_40_')) 
        # No normalizing.  
#         pool.apply_async(__main__, (path_training, path_testing, 21, 62, 5, 5, 
#                   1e-5, 10000, True, max, zero, keep_probability,
#                   os.path.join(data_set_parent, 'TwoCNN_over_sampling_'+str(keep_probability)+'_normalize_by_max_fill_by_0_stride_40_'+iname+'_as_test_'), '1'))
        useTwoLayers = False
        kernel1size = 5
        kernel2size = kernel_size
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
        CNNClass(path_training, path_testing, m, n, kernel1size, kernel2size, learning_rate, iter_num, 
                 useTwoLayers, mean, zero, keep_probability, model_dir, over_sampling_class, over_sampling_fold)
        
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
    input_dir, cross_num, fold_num, learning_rate, iter_num, keep_probability, kernel_size
                                = sys.argv[1:8]
    cross_num, fold_num, iter_num, kernel_size = int(cross_num), int(fold_num), int(iter_num), int(kernel_size)
    learning_rate, keep_probability = float(learning_rate), float(keep_probability)
    if len(sys.argv) >= 10:
        over_sampling_class, over_sampling_fold = sys.argv[8:10]
        over_sampling_fold = int(over_sampling_fold)
    else:
        over_sampling_class = None
        over_sampling_fold = 1
    generate_fake = False
    for i in range(0, cross_num):
        output_dir = os.path.join(os.path.dirname(input_dir), str(fold_num)+'xCV_data_rand_'+str(i))
        if not os.path.exists(output_dir):
            splitDataSet(input_dir, output_dir, fold_num)
            paths_mats = glob.glob(os.path.join(output_dir, 'part_*' ,'*'+MAT_SUFFIX))
            
            if generate_fake:                 
                [generateFakeMatBy(m) for m in paths_mats]
        
        crossValidate(output_dir, learning_rate, iter_num, keep_probability, kernel_size, over_sampling_class, over_sampling_fold):    
    #crossValidate()