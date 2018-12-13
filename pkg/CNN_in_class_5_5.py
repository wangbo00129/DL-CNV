import tensorflow as tf
import time
import os
from numpy import mat, row_stack
import random
import pickle
import numpy as np
from math import log
import pandas as pd
from pandas import DataFrame as df
pd.set_option('display.max_columns',10)

# Global var
DO_NORMALIZE = True
INIT_MINIMIZER_STEP_SIZE = 1e-5
PRINT_MAT = False 
REPLACE_ZEROS_WITH_MEAN = True
np.set_printoptions(threshold=np.inf)
LOG_FILE_PATH = 'z:/temp.log'
str_logs = ''
def log_i(s, f=LOG_FILE_PATH):
    global str_logs
    if str(s)=='EOF':
        with open(f, 'a+') as fw:
            fw.write(str(str_logs)+'\r\n')
        str_logs = ''
    else:        
        str_logs += str(s)+'\r\n'

class DataSet(object):
    def __init__(self, path):
        self.files=[]
        for fn in os.listdir(path):
            self.files.append(os.path.join(path, fn))
        # log_i(self.files) 
        # Shuffle. Very necessary. 
        log_i('Shuffling', LOG_FILE_PATH)
        random.shuffle(self.files)
        self.current_read=0
    
    def count(self):
        return len(self.files)
    
    def readBatch(self, batch_num=10,shuffle=False):
        if shuffle:
            random.shuffle(self.files)
        xs=mat([])
        ys=[]
        sample_names=[]
        #log_i('_current_read: ',self.current_read, LOG_FILE_PATH)
        for _ in range(batch_num):
            f=self.files[self.current_read]
            # Features
            # log_i('current file: ', f, LOG_FILE_PATH)
            current_mat = pickle.load(open(f,'rb'), encoding='iso-8859-1')
            #log_i('shape: ', LOG_FILE_PATH)
            #log_i(np.shape(xs), LOG_FILE_PATH)
            if DO_NORMALIZE:
                current_mat = current_mat/current_mat.mean()
            if REPLACE_ZEROS_WITH_MEAN:
                current_mat[current_mat==0] = 1 
            if PRINT_MAT:
                log_i(list(current_mat)) 
            
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
        return mat(xs), mat(ys), sample_names





class CNNClass(object):
    # Initialization
    def __init__(self, size_conv1, size_conv2):
        sess=tf.InteractiveSession()
        x=tf.placeholder(tf.float32, shape=[None, 50])
        
        x_2d=tf.reshape(x, [-1, 21, 50, 1])
        # y is the one-hot encoding label. 
        # 0 is negative and 1 is positive. 
        y=tf.placeholder(tf.float32, shape=[None, 2])
        
        # First conv net
        # 5, 5 is the kernel width, height. 
        # 1 is the in channel num. 
        # 32 is the out channel num. 
        w_conv1=tf.Variable(tf.truncated_normal(stddev=0.1, shape=(size_conv1, size_conv1, 1, 32)))
        b_conv1=tf.Variable(tf.constant(0.1, shape=[32]))
        conv1=tf.nn.conv2d(x_2d, w_conv1, [1, 1, 1, 1], padding='SAME')
        # A [None, 28, 28, 32] convolutional layer generated. 
        # Each conv followed by a relu.
        relu1=tf.nn.relu(conv1+b_conv1)
        # Each relu followed by max pooling. 
        out1=tf.nn.max_pool(relu1,[1,2,2,1], [1,2,2,1], padding='SAME')
        # out1 is now with shape [None, 14, 14, 32]
        
        w_conv2=tf.Variable(tf.truncated_normal(stddev=0.1, shape=[size_conv2, size_conv2, 32, 64]))
        b_conv2=tf.Variable(tf.constant(0.1, shape=[64]))
        conv2=tf.nn.conv2d(out1, w_conv2, [1, 1, 1, 1], padding='SAME')
        # A [None, 28, 28, 32] convolutional layer generated. 
        # Each conv followed by a relu.
        relu2=tf.nn.relu(conv2+b_conv2)
        # Each relu followed by max pooling. 
        out2=tf.nn.max_pool(relu2, [1,2,2,1],[1,2,2,1], padding='SAME')
        # out2 is now with shape [None, 7, 7, 64]
        
        
        # Densely connected layer. Reshape the tensor to [None, 7*7*64]. 
        flattened=tf.reshape(out2, [-1, 6*13*64])
        w=tf.Variable(tf.truncated_normal(stddev=0.1, shape=[6*13*64, 1024]))
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
        
        # Calculate loss. # Minus sign is not needed. 
        cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))

        minimizer=tf.train.AdamOptimizer(INIT_MINIMIZER_STEP_SIZE).minimize(cross_entropy)
        correct_mat=tf.cast(tf.equal(tf.argmax(y,1), tf.argmax(yhat,1)), tf.float32)
        accuracy=tf.reduce_mean(correct_mat)
        predict_mat = tf.argmax(yhat,1)
        real_positive_mat = tf.cast(tf.equal(tf.argmax(y,1), 0), tf.float32)
        real_negative_mat = tf.cast(tf.equal(tf.argmax(y,1), 1), tf.float32)
        #hit1_mat = tf.cast(tf.equal(predict_mat, true_mat), tf.float32)
        TP_mat = tf.cast(tf.multiply(correct_mat, real_positive_mat), tf.float32)
        FN_mat = tf.cast(tf.multiply(1-correct_mat, real_positive_mat), tf.float32)
        TN_mat = tf.cast(tf.multiply(correct_mat, real_negative_mat), tf.float32)
        FP_mat = tf.cast(tf.multiply(1-correct_mat, real_negative_mat), tf.float32)
        #recall_rate=tf.reduce_mean()
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        save_sess_path='d:\\eclipse-workspace\\CNN_CNV\\5_5_sess_stride_conv_size_'+str(size_conv1)+'_'+str(size_conv2)+'\\'
        run_log = os.path.join(save_sess_path, 'run_log')
        #merged = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES))
        #writer = tf.summary.FileWriter(run_log, sess.graph)
        LOG_FILE_PATH = os.path.join(save_sess_path, 'log.log')
        print('LOG_FILE_PATH changed to: '+LOG_FILE_PATH)
        
        if os.path.exists(save_sess_path):
            saver.restore(sess, save_sess_path)
        else:
            os.makedirs(save_sess_path)
            
        def generateAccuracyDataFrame(fed_xs, fed_ys, fed_names, keep_probability=1.0):
            f_dict={x:fed_xs,y:fed_ys,keep_prob:keep_probability}
            dict_test = {}
            dict_test['predict']=list(yhat.eval(feed_dict=f_dict))
            dict_test['real']=list(fed_ys)
            dict_test['correct']=list(correct_mat.eval(feed_dict=f_dict))
            dict_test['TP']=list(TP_mat.eval(feed_dict=f_dict))
            dict_test['FN']=list(FN_mat.eval(feed_dict=f_dict))
            dict_test['TN']=list(TN_mat.eval(feed_dict=f_dict))
            dict_test['FP']=list(FP_mat.eval(feed_dict=f_dict))
            accu = sum(dict_test['correct'])/len(dict_test['correct'])
            recall_rate = float(sum(dict_test['TP']))/(sum(dict_test['TP'])+sum(dict_test['FN']))
            false_negative_rate = float(sum(dict_test['FP']))/(sum(dict_test['FP'])+sum(dict_test['TN']))
            re_dataframe = df(dict_test, index=list(fed_names))
        
            return re_dataframe, accu, recall_rate, false_negative_rate
        
        
        training_set=DataSet('D:/eclipse-workspace/CNN_CNV/data/training/')
        testing_set=DataSet('D:/eclipse-workspace/CNN_CNV/data/testing/')
        # Run.

        tick = time.time()
        for step in range(30000):
            train_xs, train_ys, _ = training_set.readBatch(20, True)
            train_feed_dict = {x:train_xs,y:train_ys,keep_prob:0.6}
            
            #result = minimizer.run(feed_dict=train_feed_dict) 
            
            result = minimizer.run(feed_dict=train_feed_dict)
            if not step%100 == 0:
                continue
            
            log_i('training', LOG_FILE_PATH)
            accu = accuracy.eval(feed_dict={x:train_xs,y:train_ys,keep_prob:1.0})
            log_i('training accuracy: '+str(accu), LOG_FILE_PATH)
            
            #log_i(yhat.eval(feed_dict={x:train_xs,y:train_ys,keep_prob:1.0}), LOG_FILE_PATH)
            log_i('training cross entropy: ', LOG_FILE_PATH)
            log_i(cross_entropy.eval(feed_dict={x:train_xs,y:train_ys,keep_prob:1.0}), LOG_FILE_PATH)
            
            log_i('testing', LOG_FILE_PATH)
            test_xs, test_ys, test_sample_names = testing_set.readBatch(testing_set.count())
            
            df_accu, accu, recall_rate, false_negative_rate = generateAccuracyDataFrame(test_xs, test_ys, test_sample_names)
            
            log_i(df_accu, LOG_FILE_PATH)
            log_i('testing accuracy: '+str(accu), LOG_FILE_PATH)
            log_i('testing recall_rate: '+str(recall_rate), LOG_FILE_PATH)
            log_i('testing false positive: '+str(false_negative_rate), LOG_FILE_PATH)
            log_i('batch done', LOG_FILE_PATH)
            saver.save(sess, save_path=save_sess_path)
            #writer.add_summary(minimizer, step)
        #     
        #     log_i(accuracy.eval(feed_dict={x:test_xs,y:test_ys,keep_prob:1.0}))    
        #     dict_test = {}
        #     dict_test['name']=list(test_sample_names)
        #     dict_test['predict']=list(yhat.eval(feed_dict={x:test_xs,y:test_ys,keep_prob:1.0}))
        #     dict_test['real']=list(test_ys)
        #     log_i(df(dict_test), LOG_FILE_PATH)
            
            
        
        tock = time.time()
        log_i('time elapsed: ', LOG_FILE_PATH)
        log_i(tock - tick, LOG_FILE_PATH)
        log_i('EOF', LOG_FILE_PATH)
        sess.close()
            
    
   


#testing_set=DataSet('D:/CNN_CNV/data/strong_testing/')
#log_i(training_set.readBatch(15), LOG_FILE_PATH)
#log_i('__', LOG_FILE_PATH)
#log_i(testing_set.readBatch(10), LOG_FILE_PATH)

    

def __main__(conv1_size, conv2_size):
    cnn_class = CNNClass(conv1_size, conv2_size)
        
if __name__=='__main__':
    for conv1 in range(5, 6):
        for conv2 in range(5, 6):
            __main__(conv1, conv2)