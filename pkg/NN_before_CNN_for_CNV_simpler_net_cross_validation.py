import tensorflow as tf
import time
import os
import glob
from numpy import mat, row_stack, mean
import random
import pickle
import numpy as np
from math import log
import pandas as pd
from pandas import DataFrame as df 
pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',500)
pd.set_option('display.width',1000)
np.set_printoptions(threshold=np.inf)
from pkg.DataSet import DataSet
# Global var
PRINT_MAT = False 
class CNNClass(object):
    # Initialization
    def log_i2(self, s, f):
        if str(s)=='EOF':
            with open(f, 'a+') as fw:
                fw.write(str(self.str_logs)+'\n')
            self.str_logs = ''
        else:        
            self.str_logs += str(s)+'\n'
    
    def log_i(self, s, f):
        if not str(s)=='EOF':
            with open(f, 'a+') as fw:
                fw.write(str(s)+'\n')
        
    def __init__(self, path_training_set, path_testing_set, mat_shape_m, mat_shape_n, size_conv1, size_conv2, 
                 INIT_MINIMIZER_STEP_SIZE = 1e-5, step_num = 10000, useTwoCNN=False, 
                 DO_NORMALIZE_BY = mean, REPLACE_ZEROS_WITH_SOME_VALUE = mean, keep_probability=0.6, 
                 output_prefix = 'd:\\eclipse-workspace\\CNN_CNV\\rewrite_stride_40_save_sess_stride_conv_size_', over_sampling='1'):
            # Over sample the class 1 as default since the samples in class 1 are less. 
        save_sess_path=output_prefix+str(size_conv1)+'_'+str(size_conv2)+'\\'
        run_log = os.path.join(save_sess_path, 'run_log')
        LOG_FILE_PATH = os.path.join(save_sess_path, 'log.log')
        
        self.str_logs = ''
        self.g = tf.Graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess=tf.Session(graph=self.g, config=config)
        with self.sess.as_default():
            with self.g.as_default():             
            #sess=tf.Session()
                x=tf.placeholder(tf.float32, shape=[None, mat_shape_n])
                flatten_x=tf.reshape(x, shape=[-1, mat_shape_m*mat_shape_n])
                
                mean_of_matrix = tf.reduce_mean(flatten_x, axis=1) # aixs=1 means the mean of every matrix. 
                mean_of_matrix = tf.reshape(mean_of_matrix, shape=[-1, 1])
                weights_for_mean = tf.Variable(tf.truncated_normal([1,2])) 
                bias_for_mean = tf.Variable(tf.zeros([2]))
                output_mean = tf.matmul(mean_of_matrix, weights_for_mean)
                output_mean = output_mean+bias_for_mean
                output_mean = tf.nn.relu(output_mean)
                # Later will be used. 
                #flatten_x=tf.matmul(1/tf.transpose(tf.reduce_mean(flatten_x, 1, keepdims=True)), flatten_x)
                flatten_x=flatten_x/tf.tile(tf.reduce_mean(flatten_x, 1, keepdims=True),  [1, mat_shape_m*mat_shape_n])
                x_2d = tf.reshape(flatten_x, [-1, mat_shape_m, mat_shape_n, 1]) 
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
                # A [None, 28, 28, 32] convolutional layer generated. 
                # Each conv followed by a relu.
                # Each relu followed by max pooling. 
                # out2 is now with shape [None, 7, 7, 64]

                if useTwoCNN: 
                    w_conv2=tf.Variable(tf.truncated_normal(stddev=0.1, shape=[size_conv2, size_conv2, 32, 64]))
                    b_conv2=tf.Variable(tf.constant(0.1, shape=[64]))
                    conv2=tf.nn.conv2d(out1, w_conv2, [1, 1, 1, 1], padding='SAME')
                    # A [None, 28, 28, 32] convolutional layer generated. 
                    # Each conv followed by a relu.
                    relu2=tf.nn.relu(conv2+b_conv2)
                    # Each relu followed by max pooling. 
                    out2=tf.nn.max_pool(relu2, [1,2,2,1],[1,2,2,1], padding='SAME')
                    # out2 is now with shape [None, 7, 7, 64]
                    final_mat_m = int(np.ceil(mat_shape_m/4))
                    final_mat_n = int(np.ceil(mat_shape_n/4))
                    flattened=tf.reshape(out2, [-1, final_mat_m*final_mat_n*64])
                    w=tf.Variable(tf.truncated_normal(stddev=0.1, shape=[final_mat_m*final_mat_n*64, 1024]))
                    b=tf.Variable(tf.truncated_normal(stddev=0.1, shape=[1024]))
                else: 
                    final_mat_m = int(np.ceil(mat_shape_m/2))
                    final_mat_n = int(np.ceil(mat_shape_n/2))
                    flattened=tf.reshape(out1, [-1, final_mat_m*final_mat_n*32])
                    # Densely connected layer. Reshape the tensor to [None, 7*7*64]. 
                    w=tf.Variable(tf.truncated_normal(stddev=0.1, shape=[final_mat_m*final_mat_n*32, 1024]))
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
                relu4 = relu4
                
                # Add a two by two neural network. 
                merged = tf.concat([mean_of_matrix, relu4], axis=1, name='merged')
                wei=tf.Variable(tf.truncated_normal(stddev=0.1, shape=[3,2]))
                bia=tf.Variable(tf.truncated_normal(stddev=0.1, shape=[2]))
                mul=tf.matmul(merged, wei)+bia
                relu5=tf.nn.relu(mul)
                wei2=tf.Variable(tf.truncated_normal(stddev=0.1, shape=[2,2]))
                bia2=tf.Variable(tf.truncated_normal(stddev=0.1, shape=[2]))
                relu6=tf.nn.relu(mul)
                yhat=tf.nn.softmax(relu6)
                
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
                initializer = tf.global_variables_initializer()
                
                initializer.run()
                saver = tf.train.Saver()
                summary_writer = tf.summary.FileWriter(save_sess_path, self.sess.graph)
        #merged = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES))
        #writer = tf.summary.FileWriter(run_log, sess.graph)

            
                if os.path.exists(save_sess_path):
                    if len(os.listdir(save_sess_path))>1:
                        saver.restore(self.sess, save_sess_path)
                else:
                    os.makedirs(save_sess_path, exist_ok=True)
                    
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
                    re_dataframe = df(dict_test, index=list(fed_names), columns=['predict', 'real', 'correct', 'TP', 'FN', 'TN', 'FP'])
                
                    return re_dataframe, accu, recall_rate, false_negative_rate
            
        
                training_set=DataSet(path_training_set, DO_NORMALIZE_BY, REPLACE_ZEROS_WITH_SOME_VALUE, over_sampling)
                testing_set=DataSet(path_testing_set, DO_NORMALIZE_BY, REPLACE_ZEROS_WITH_SOME_VALUE)
                # Run.
                
                tick = time.time()
                for step in range(1, step_num+1):
                    train_xs, train_ys, train_names = training_set.readBatch(20, True)
                    train_feed_dict = {x:train_xs,y:train_ys,keep_prob:keep_probability}
                    
                    #result = minimizer.run(feed_dict=train_feed_dict) 
                    
                    result = minimizer.run(feed_dict=train_feed_dict)
                    if not step%100 == 0:
                        continue
                    
                    self.log_i('training', LOG_FILE_PATH)
                    df_accu, accu, recall_rate, false_negative_rate = generateAccuracyDataFrame(train_xs, train_ys, train_names)
                    #self.log_i(df_accu, LOG_FILE_PATH)
                    accu = accuracy.eval(feed_dict={x:train_xs,y:train_ys,keep_prob:1.0})
                    self.log_i('training accuracy: '+str(accu), LOG_FILE_PATH)
                    print(self.sess.run(merged, feed_dict={x:train_xs,y:train_ys,keep_prob:1.0}))
                    #self.log_i(yhat.eval(feed_dict={x:train_xs,y:train_ys,keep_prob:1.0}), LOG_FILE_PATH)
                    self.log_i('training cross entropy: ', LOG_FILE_PATH)
                    self.log_i(cross_entropy.eval(feed_dict={x:train_xs,y:train_ys,keep_prob:1.0}), LOG_FILE_PATH)
                    
                    self.log_i('testing', LOG_FILE_PATH)
                    test_xs, test_ys, test_sample_names = testing_set.readBatch(testing_set.count())
                    
                    df_accu, accu, recall_rate, false_negative_rate = generateAccuracyDataFrame(test_xs, test_ys, test_sample_names)
                    
                    self.log_i(df_accu, LOG_FILE_PATH)
                    self.log_i('testing accuracy: '+str(accu), LOG_FILE_PATH)
                    self.log_i('testing recall_rate: '+str(recall_rate), LOG_FILE_PATH)
                    self.log_i('testing false positive: '+str(false_negative_rate), LOG_FILE_PATH)
                    self.log_i(str(step)+' steps done', LOG_FILE_PATH)
                    saver.save(self.sess, save_sess_path, step)
                    #writer.add_summary(minimizer, step)
                #     
                #     self.log_i(accuracy.eval(feed_dict={x:test_xs,y:test_ys,keep_prob:1.0}))    
                #     dict_test = {}
                #     dict_test['name']=list(test_sample_names)
                #     dict_test['predict']=list(yhat.eval(feed_dict={x:test_xs,y:test_ys,keep_prob:1.0}))
                #     dict_test['real']=list(test_ys)
                #     self.log_i(df(dict_test), LOG_FILE_PATH)
                                
                tock = time.time()
                self.log_i('time elapsed: ', LOG_FILE_PATH)
                self.log_i(tock - tick, LOG_FILE_PATH)
                self.log_i('EOF', LOG_FILE_PATH)
                self.sess.close()

#testing_set=DataSet('D:/CNN_CNV/data/strong_testing/')
#self.log_i(training_set.readBatch(15), LOG_FILE_PATH)
#self.log_i('__', LOG_FILE_PATH)
#self.log_i(testing_set.readBatch(10), LOG_FILE_PATH)

    

# 
# def __main__(path_training_set, path_testing_set, mat_shape_m, mat_shape_n, step_size, step_num, conv1_size, conv2_size, useTwoCNN, 
#              DO_NORMALIZE_BY, REPLACE_ZEROS_WITH_SOME_VALUE, keep_probability, output_prefix, over_sampling=None):

def __main__(*args):
    # Paras: 
    # DO_NORMALIZE_BY: while normalizing, divide by the num generated by this function. 
    # REPLACE_ZEROS_WITH_SOME_VALUE: while replacing 0s, replace 0s with the num generated by this function 
#     path_training_set = 'D:/eclipse-workspace/CNN_CNV/data_stride_40/training/'
#     path_testing_set = 'D:/eclipse-workspace/CNN_CNV/data_stride_40/testing/'
#     cnn_class = CNNClass(path_training_set, path_testing_set, mat_shape_m, mat_shape_n, conv1_size, conv2_size, 
#                          step_size, step_num, useTwoCNN, DO_NORMALIZE_BY, REPLACE_ZEROS_WITH_SOME_VALUE, keep_probability, output_prefix, over_sampling)

    cnn_class = CNNClass(*args)
#     cnn_class = CNNClass(path_training_set, path_testing_set, 21, 62, conv1_size, conv2_size, 
#                          1e-5, 30000, mean, mean, 
#                          output_prefix='d:/eclipse-workspace/CNN_CNV/rewrite_para3_no_self_stride_40_save_sess_stride_conv_size_')
#             
if __name__=='__main__':
    for conv1 in range(5,6):
        for conv2 in range(5,6):
            __main__(conv1, conv2)