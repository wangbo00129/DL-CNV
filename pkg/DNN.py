import tensorflow as tf
import os
from numpy import mat, row_stack
import random
import pickle
import sys
import numpy as np


# Global var
DO_NORMALIZE = True
MINIMIZER_STEP_SIZE = 3e-4
PRINT_MAT = False 
REPLACE_ZEROS_WITH_MEAN = True
np.set_printoptions(threshold=np.inf)

# Initialization
sess=tf.InteractiveSession()
x=tf.placeholder(tf.float32, shape=[None, 50])

x_2d=tf.reshape(x, [-1, 21, 50, 1])
# y is the one-hot encoding label. 
# 0 is negative and 1 is positive. 
y=tf.placeholder(tf.float32, shape=[None, 2])

# Output layer. 
w2=tf.Variable(tf.truncated_normal(stddev=0.1, shape=[1024, 2]))
b2=tf.Variable(tf.truncated_normal(stddev=0.1, shape=[2]))
relu4=tf.nn.relu(tf.matmul(relu3_dropped, w2)+b2)
yhat=tf.nn.softmax(relu4)

# Calculate loss.
# Minus sign is not needed. 
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))


correct_mat=tf.cast(tf.equal(tf.argmax(y,1), tf.argmax(yhat,1)), tf.float32)
accuracy=tf.reduce_mean(correct_mat)
tf.global_variables_initializer().run()

# Run.
class DataSet(object):
    def __init__(self, path):
        self.files=[]
        for fn in os.listdir(path):
            self.files.append(os.path.join(path, fn))
        # print(self.files) 
        # Shuffle. Very necessary. 
        print('Shuffling')
        random.shuffle(self.files)
        self.current_read=0
    
    def readBatch(self, batch_num=10):
        xs=mat([])
        ys=[]
        #print('_current_read: ',self.current_read)
        for i in range(batch_num):
            f=self.files[self.current_read]
            # Features
            # print('current file: ', f)
            current_mat = pickle.load(open(f,'rb'))
            #print('shape: ')
            #print(np.shape(xs))
            if DO_NORMALIZE:
                current_mat = current_mat/current_mat.mean()
            if REPLACE_ZEROS_WITH_MEAN:
                current_mat[current_mat==0] = 1 
            if PRINT_MAT:
                print(list(current_mat)) 
            
            if not np.shape(xs)[1]==0:
                xs = row_stack((xs, current_mat))
            else:
                xs = current_mat
            # One-hot labels
            y_init=[0,0]
            index=int(os.path.basename(f).split('_')[0])
            y_init[index-1]=1
            ys.append(y_init)
            if self.current_read < len(self.files)-1:
                self.current_read+=1
            else:
                self.current_read=0
        return mat(xs), mat(ys)
        
training_set=DataSet('/data2/wangb/CNN_CNV/data/training/')
testing_set=DataSet('/data2/wangb/CNN_CNV/data/testing/')

#print(training_set.readBatch(15))
#print('__')
#print(testing_set.readBatch(10))

for step in range(20000):
    train_xs, train_ys = training_set.readBatch(21)
    #print('final shape: ')
    #print(np.shape(train_xs))
    minimizer.run(feed_dict={x:train_xs,y:train_ys,keep_prob:0.5})
    if not step%100 == 0:
        continue
    print('training')
    print(accuracy.eval(feed_dict={x:train_xs,y:train_ys,keep_prob:1.0}))
    test_xs, test_ys = testing_set.readBatch(10)
    print(yhat.eval(feed_dict={x:train_xs,y:train_ys,keep_prob:1.0}))
    print('cross entropy: ')
    print(cross_entropy.eval(feed_dict={x:train_xs,y:train_ys,keep_prob:1.0}))
    print('testing')
    print(accuracy.eval(feed_dict={x:test_xs,y:test_ys,keep_prob:1.0}))
    print('predict')
    print(yhat.eval(feed_dict={x:test_xs,y:test_ys,keep_prob:1.0}))
    print('real')
    print(test_ys)

