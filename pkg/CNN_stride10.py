import tensorflow as tf
import os
from numpy import mat, row_stack
import random
import pickle
import sys
import numpy as np
from math import log

# Global var
DO_NORMALIZE = True
INIT_MINIMIZER_STEP_SIZE = 1e-5
PRINT_MAT = False 
REPLACE_ZEROS_WITH_MEAN = True
np.set_printoptions(threshold=np.inf)

# Initialization
sess=tf.InteractiveSession()
x=tf.placeholder(tf.float32, shape=[None, 246])

x_2d=tf.reshape(x, [-1, 21, 246, 1])
# y is the one-hot encoding label. 
# 0 is negative and 1 is positive. 
y=tf.placeholder(tf.float32, shape=[None, 2])

# First conv net
# 5, 5 is the kernel width, height. 
# 1 is the in channel num. 
# 32 is the out channel num. 
w_conv1=tf.Variable(tf.truncated_normal(stddev=0.1, shape=(5, 5, 1, 32)))
b_conv1=tf.Variable(tf.constant(0.1, shape=[32]))
conv1=tf.nn.conv2d(x_2d, w_conv1, [1, 1, 1, 1], padding='SAME')
# A [None, 28, 28, 32] convolutional layer generated. 
# Each conv followed by a relu.
relu1=tf.nn.relu(conv1+b_conv1)
# Each relu followed by max pooling. 
out1=tf.nn.max_pool(relu1,[1,2,2,1], [1,2,2,1], padding='SAME')
# out1 is now with shape [None, 14, 14, 32]

w_conv2=tf.Variable(tf.truncated_normal(stddev=0.1, shape=[5, 5, 32, 64]))
b_conv2=tf.Variable(tf.constant(0.1, shape=[64]))
conv2=tf.nn.conv2d(out1, w_conv2, [1, 1, 1, 1], padding='SAME')
# A [None, 28, 28, 32] convolutional layer generated. 
# Each conv followed by a relu.
relu2=tf.nn.relu(conv2+b_conv2)
# Each relu followed by max pooling. 
out2=tf.nn.max_pool(relu2, [1,2,2,1],[1,2,2,1], padding='SAME')
# out2 is now with shape [None, 7, 7, 64]


# Densely connected layer. Reshape the tensor to [None, 7*7*64]. 
flattened=tf.reshape(out2, [-1, 6*62*64])
w=tf.Variable(tf.truncated_normal(stddev=0.1, shape=[6*62*64, 1024]))
b=tf.Variable(tf.truncated_normal(stddev=0.1, shape=[1024]))
m1=tf.matmul(flattened, w)+b
relu3=tf.nn.relu(m1)

# Dropout layer. 
keep_prob=tf.placeholder(tf.float32)
relu3_dropped=tf.nn.dropout(relu3, keep_prob)

# Output layer. 
w2=tf.Variable(tf.truncated_normal(stddev=0.1, shape=[1024, 2]))
b2=tf.Variable(tf.truncated_normal(stddev=0.1, shape=[2]))
relu4=tf.nn.relu(tf.matmul(relu3_dropped, w2)+b2)
yhat=tf.nn.softmax(relu4)

# Calculate loss.
# Minus sign is not needed. 
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
minimizer_step = tf.placeholder(tf.float32)
minimizer=tf.train.AdamOptimizer(minimizer_step).minimize(cross_entropy)
correct_mat=tf.cast(tf.equal(tf.argmax(y,1), tf.argmax(yhat,1)), tf.float32)
accuracy=tf.reduce_mean(correct_mat)
predict_mat = tf.argmax(yhat,1)
true_mat = tf.argmax(y,1)
hit1_mat = tf.cast(tf.equal(predict_mat, true_mat), tf.float32)

#recall_rate=tf.reduce_mean()
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
    
    def readBatch(self, batch_num=10,shuffle=False):
        if shuffle:
            random.shuffle(self.files)
        xs=mat([])
        ys=[]
        sample_names=[]
        #print('_current_read: ',self.current_read)
        for i in range(batch_num):
            f=self.files[self.current_read]
            # Features
            # print('current file: ', f)
            current_mat = pickle.load(open(f,'rb'), encoding='iso-8859-1')
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
            sample_names.append(os.path.basename(f).split('_')[1].replace('.mat',''))
            if self.current_read < len(self.files)-1:
                self.current_read+=1
            else:
                self.current_read=0
        return mat(xs), mat(ys), mat(sample_names)
        
training_set=DataSet('D:/CNN_CNV/data_stride_10/training/')
testing_set=DataSet('D:/CNN_CNV/data_stride_10/testing/')

#print(training_set.readBatch(15))
#print('__')
#print(testing_set.readBatch(10))
import time
tick = time.time()
for step in range(200000):
    train_xs, train_ys, _ = training_set.readBatch(20, True)
    #print('final shape: ')
    #print(np.shape(train_xs))
    minimizer.run(feed_dict={x:train_xs,y:train_ys,keep_prob:0.5,minimizer_step:INIT_MINIMIZER_STEP_SIZE/log(step+3)})
    if not step%100 == 0:
        continue
    print('training')
    print(accuracy.eval(feed_dict={x:train_xs,y:train_ys,keep_prob:1.0}))
    test_xs, test_ys, test_sample_names = testing_set.readBatch(10)
    print(yhat.eval(feed_dict={x:train_xs,y:train_ys,keep_prob:1.0}))
    print('cross entropy: ')
    print(cross_entropy.eval(feed_dict={x:train_xs,y:train_ys,keep_prob:1.0}))
    print('testing')
    print(accuracy.eval(feed_dict={x:test_xs,y:test_ys,keep_prob:1.0}))
    print('sample')
    print(test_sample_names)
    print('predict')
    print(yhat.eval(feed_dict={x:test_xs,y:test_ys,keep_prob:1.0}))
    print('real')
    print(test_ys)
tock = time.time()
print('time elapsed: ')
print(tock - tick)
