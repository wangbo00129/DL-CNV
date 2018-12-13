'''

@author: dell
'''

import os
import pickle 
import glob

def readMat(mat_path):
    if not os.path.exists(mat_path):
        return 
    with open(mat_path,'rb') as fr:
        current_mat = pickle.load(fr, encoding='iso-8859-1')
        return current_mat

def strideMore(mat_in, n_stride=2):
    # stride every n_stride original stride 
    
    return mat_in[:,::n_stride]

def strideMoreInFolder(folder_in, folder_out, n_stride):
    list_in = glob.glob(os.path.join(folder_in, '*'))
    for i in range(len(list_in)):
        list_in[i]=list_in[i].replace(folder_in, folder_out)
        os.makedirs(list_in[i], exist_ok=True)
    list_in = glob.glob(os.path.join(folder_in, '*', '*.mat'))
    for m in list_in:
        mat_in = readMat(m)
        mat_out = strideMore(mat_in, n_stride) 
        path_mat_out = m.replace(folder_in, folder_out)
        with open(path_mat_out, 'wb') as fw:
            pickle.dump(mat_out, fw)
        
if __name__ == '__main__':
    parent_mat_in = 'D:\eclipse-workspace\CNN_CNV\data_producing_soft_links_mats_stride_10'
    parent_mat_out = 'D:\eclipse-workspace\CNN_CNV\data_producing_soft_links_mats_stride_40'
    parent_mat_in = 'Z:/neg'
    parent_mat_out = 'Z:/neg_40'
    parent_mat_in = "D:/eclipse-workspace/CNN_CNV/data_single_end_model/stride10"
    parent_mat_out = "D:/eclipse-workspace/CNN_CNV/data_single_end_model/mixed_stride40"
    strideMoreInFolder(parent_mat_in, parent_mat_out, 4)