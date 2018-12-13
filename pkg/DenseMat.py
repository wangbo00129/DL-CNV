'''
Created on 20181026


@author: dell

'''
from numpy import mat, reshape
import pickle
import glob
import os

from pkg.EnumerateRefBins import RefGenomeReader 

def denseMat(path_mat_pickle):
    # 12 non zero elements in each mat. 
    non_zero_mat = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), 
                    (1, 9), (1, 10), (1, 11), (1, 12), (1, 13), (1, 14), (1, 15), (1, 16), (1, 17), (1, 18), (1, 19), (1, 20), 
                    (1, 21), (1, 22), (1, 23), (1, 24), (1, 25), (1, 26), (1, 27), (1, 28), (1, 29), (2, 0), (2, 1), (2, 2), 
                    (2, 3), (3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2), (4, 3), (5, 0), (5, 1), (5, 2), (6, 0), (6, 1), 
                    (7, 0), (7, 1), (7, 2), (8, 0), (8, 1), (8, 2), (9, 0), (9, 1), (10, 0), (10, 1), (10, 2), (10, 3), 
                    (10, 4), (11, 0), (11, 1), (11, 2), (12, 0), (12, 1), (12, 2), (13, 0), (13, 1), (13, 2), (14, 0), 
                    (14, 1), (14, 2), (14, 3), (14, 4), (15, 0), (16, 0), (16, 1), (16, 2), (16, 3), (17, 0), (17, 1), 
                    (18, 0), (18, 1), (18, 2), (19, 0), (19, 1), (19, 2), (20, 0), (20, 1), (20, 2), (20, 3), (20, 4), 
                    (20, 5), (20, 6), (20, 7), (20, 8), (20, 9), (20, 10), (20, 11), (20, 12), (20, 13), (20, 14), (20, 15), 
                    (20, 16), (20, 17), (20, 18), (20, 19), (20, 20), (20, 21), (20, 22), (20, 23), (20, 24), (20, 25), (20, 26), 
                    (20, 27), (20, 28), (20, 29), (20, 30), (20, 31), (20, 32), (20, 33), (20, 34), (20, 35), (20, 36), (20, 37), 
                    (20, 38), (20, 39), (20, 40), (20, 41), (20, 42), (20, 43), (20, 44), (20, 45), (20, 46), (20, 47), (20, 48), 
                    (20, 49), (20, 50), (20, 51), (20, 52), (20, 53), (20, 54), (20, 55), (20, 56), (20, 57), (20, 58), (20, 59), 
                    (20, 60), (20, 61)]
    with open(path_mat_pickle, 'rb') as fr:
        in_mat = pickle.load(fr)
        out_mat = mat([in_mat[x] for x in non_zero_mat]+[0]*4)
        out_mat = reshape(out_mat, (12,13))
    return out_mat
    # Add 4 zeros to make it 12*13. 
    
    
 
# Find non-zero elements in mats according to the reference genome. 
def getValidPositionInMat(): 
    ref = RefGenomeReader('z:/pure.seq')
    ls_ls = ref.getAllBins(50, 40)
    for i in range(len(ls_ls)):
        ls = ls_ls[i]
        for j in range(len(ls)):
            print((i,j))
if __name__=='__main__':
    folder_in = "D:/eclipse-workspace/CNN_CNV/data_producing_soft_links_mats_stride_40/correct_label"
    folder_out = "D:/eclipse-workspace/CNN_CNV/data_producing_soft_links_mats_stride_40/correct_label_dense"
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)
    for i in glob.glob(folder_in+'/*mat'):
        mat_out = denseMat(i)
        with open(folder_out+'/'+os.path.basename(i),'wb') as fw:
            pickle.dump(mat_out, fw)
            