'''
Created on 20180917

@author: dell
'''
import glob
import random
import os
import pickle
from numpy import row_stack, mat, median, mean
import numpy as np

FILE_SIZE_IF_ABSENT = 500
    

PRINT_MAT = True 
class DataSet(object):
    def __init__(self, paths, DO_NORMALIZE_BY, REPLACE_ZEROS_WITH_SOME_VALUE, over_sampling=None, over_sampling_fold=20, load_fake=True):
        self.paths = paths
        self.DO_NORMALIZE_BY=DO_NORMALIZE_BY
        self.REPLACE_ZEROS_WITH_SOME_VALUE = REPLACE_ZEROS_WITH_SOME_VALUE
        self.file_sizes = {'18HE22933F_S49':111,'18LN70052F_S17':124,'18N00451F_S38':168,'18N00512F_S22':168,'18N00628F_S16':136,'18N00778F_S7':113,'18N00808F_S19':63,'18N00810H_S22':105,'18N00813F-II_S9':162,'18N00821F_S31':125,'18N00847F_S52':122,'18N00874F_S11':161,'18N00877F_S18':88,'18N00888F_S10':115,'18N00921F_S22':359,'18N00944F_S4':165,'18N01013F_S20':162,'18CF90107B_S1':196,'18CF90108B_S37':168,'18CF90111P_S9':369,'18CF90112B_S9':169,'18HE24451F_S48':403,'18HE24482QC_S3':260,'18HE24532F_S43':181,'18HE24582F_S10':244,'18HE24585F_S44':173,'18HE24601F_S16':158,'18HE24618F_S17':130,'18HE24626F_S5':156,'18HE24641P_S3':379,'18JM90036P_S6':512,'18JM90037F_S39':170,'18N01727F-_S3':184,'18N01749P_S4':511,'18N01750F_S11':427,'18N01753F_S42':176,'18N01756F_S1':144,'18N01760P_S2':379,'18N01761P_S4':371,'18N01762P_S5':370,'18N01763P_S7':494,'18N01767P_S13':412,'18XZ51280P_S7':514,'18XZ51300P_S6':397,'18ZN00050F_S6':137,'18ZN00058F_S14':148,'18ZN00059F_S19':153,'18ZN00060P_S10':400,'18ZN00061P_S11':389,'ZK0820-g_S38':160}
        tmp = ''
        with open(r'D:\eclipse-workspace\CNN_CNV\data_ERBB2_stride40_split_to_training_test\file_size_per_file.txt') as fr:
            for line in fr:
                tmp += line.strip()
        self.file_sizes = eval(tmp) 
        if type(paths) == type(''):
            paths = [paths]
        self.files=[]
        for p in paths:

            files_mat = glob.glob(os.path.join(p, '*'))
            if not load_fake:
                # Count how many '_' are in p. If >=3, it's fake. 
                for m in files_mat:                    
                    #print(os.path.basename(m).split('_'))
                    if len(os.path.basename(m).split('_'))>=4:
                        #print('continue')
                        continue
                    self.files.append(m)
            else:
                self.files.extend(files_mat)
        # self.log_i(self.files) 
        # Shuffle. Very necessary. 
        if over_sampling:
#             print('judging prefix: '+str(over_sampling)+'_') 
#             for x in self.files:
#                 
#                 tof = os.path.basename(x).startswith(str(over_sampling)+'_')
#                 print(tof)
            to_over_sample = list(filter(lambda x:os.path.basename(x).startswith(str(over_sampling)+'_'), self.files))
            to_over_sample *= (over_sampling_fold-1)
            
            self.files.extend(to_over_sample)
        print(self.paths)    
        print(len(self.files))
        random.shuffle(self.files)
        self.sample_names=[]
        self.labels={}
        self.in_mats={}
        for f in self.files:
            sample_name='_'.join(os.path.basename(f).split('_')[1:]).replace('.mat','')
            self.sample_names.append(sample_name)
            # Features
            # self.log_i('current file: ', f, LOG_FILE_PATH)
            try:
                current_mat = pickle.load(open(f,'rb'))
            except:
                current_mat = pickle.load(open(f,'rb'), encoding='iso-8859-1')
            #self.log_i('shape: ', LOG_FILE_PATH)
            #self.log_i(np.shape(xs), LOG_FILE_PATH)
            
            if self.DO_NORMALIZE_BY:
                #print(current_mat[current_mat>0])
                if self.DO_NORMALIZE_BY == 'file_size':
                    if sample_name in  self.file_sizes:
                        normalize_by = self.file_sizes[sample_name]
                    else:
                        normalize_by = FILE_SIZE_IF_ABSENT
                        print('file size NA: '+sample_name)

                else:
                    normalize_by = self.DO_NORMALIZE_BY(current_mat[current_mat>0])
                
            if self.REPLACE_ZEROS_WITH_SOME_VALUE:
                current_mat[current_mat==0] = self.REPLACE_ZEROS_WITH_SOME_VALUE(current_mat[current_mat>0])
            
            if self.DO_NORMALIZE_BY:
                current_mat = current_mat/normalize_by
            
            # print(current_mat)
            self.in_mats[sample_name] = current_mat
            # One-hot labels
            y_init=[0,0]
            index=int(os.path.basename(f).split('_')[0])
            y_init[index-1]=1
            self.labels[sample_name] = y_init
            
    
    
    def count(self):
        return len(self.sample_names)
    
    def readBatch(self, batch_num=10,shuffle=False):
        if batch_num > self.count():
            # If the batch num exceeds the training sample num. 
            snames = np.tile(self.sample_names, int(np.ceil(batch_num/self.count())))
        else:
            snames = self.sample_names
        if shuffle:
            random.shuffle(snames)
        
        snames = self.sample_names[0:batch_num]
        
        xs = [self.in_mats[x] for x in snames]
        ys = [self.labels[x] for x in snames]
        #[print(shape(x)) for x in xs]
        xs = row_stack(xs)
        #self.log_i('_current_read: ',self.current_read, LOG_FILE_PATH)
        return mat(xs), mat(ys), snames

if __name__ == '__main__':
    def zero(num):
        return 0.0
    all=DataSet('D:\\eclipse-workspace\\CNN_CNV\\data_stride_40\\all', 'file_size', zero, '1')
    for name in all.sample_names: 
        #print(name, mean(all.in_mats[name][all.in_mats[name]>0]),all.labels[name], sep='\t')
        print(name, mean(all.in_mats[name]),all.labels[name], sep='\t')
    