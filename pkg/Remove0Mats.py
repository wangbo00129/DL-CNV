'''
@author: dell
'''
from numpy import mean
import pickle
import glob
import os

parent = 'Z:\\realpos/'
files = glob.glob(parent+'/*')
def readMatFromPickle(path):
    return pickle.load(open(path, 'rb'), encoding = 'iso-8859-1')

for f in files:
    if mean(readMatFromPickle(f))==0:
        os.remove(f)