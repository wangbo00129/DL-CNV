'''
@author: dell
'''
from numpy import mean
import pickle
import glob
import os

parent_reference = 'D:/eclipse-workspace/CNN_CNV/data_large_model/mixed/'
parent = 'z:/realpos//'

def renameFileAccordingTo(file, reference_path):
    fn = os.path.basename(file)
    sample_name = fn[2:].replace('.mat', '')
    #reference_file = glob.glob(os.path.join(reference_path, '*_'+sample_name+'.mat'))[0]
    file_num = len(glob.glob(os.path.join(reference_path, '*'+sample_name+'*join.mat')))
    #print(os.path.join(reference_path, '*_'+sample_name+'.join.mat'))
    #print(file_num)
    if not file_num == 1:
        print(sample_name)
    #to_name = os.path.basename(reference_file).replace('.mat', '.join.mat')
    
    #os.rename(file, os.path.join(os.path.dirname(file), to_name))
#     print(file)
#     print('to: ')
#     print(os.path.join(os.path.dirname(file), to_name))
    
    
def removeNotShownBefore(currentdir, olddir):
    currentall = os.listdir(currentdir)
    oldall = os.listdir(olddir)
    currentall = [x.replace('.mat', '.join.mat') for x in currentall]
    for x in currentall:
        if x not in oldall:
            print('rm '+x)
    
    
if __name__=='__main__':
#     for f in glob.glob(parent_reference+"/*"):
#         
#         renameFileAccordingTo(f, parent)
    removeNotShownBefore(parent_reference, parent)   