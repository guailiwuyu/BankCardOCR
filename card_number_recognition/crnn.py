import os
import time
import numpy as np
import tensorflow as tf
import config
from scipy.misc import imread, imresize, imsave
from tensorflow.contrib import rnn
import tensorflow.contrib.slim as slim

from data_manager import DataManager
from utils import sparse_tuple_from, resize_image, label_to_array, ground_truth_to_word, levenshtein

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class CRNN(object):
    def __init__(self,iteration_count, batch_size, model_path, examples_path, max_image_width, train_test_ratio, restore,isTrain):
        
        self.global_step = tf.Variable(0,name='global_step')
        self.batch_size = batch_size
        self.step = 0
        self.__model_path = model_path
        self.__save_path = os.path.join(model_path, 'ckp')

        self.__restore = restore
        
        self.__training_name = str(int(time.time()))
        self.__session = tf.Session()
        self.isTrain = isTrain
        #tensorboard
        self.train_writer = tf.summary.FileWriter('logs/train')
        self.test_writer = tf.summary.FileWriter('logs/test')
        
        # Building graph
        with self.__session.as_default():
            (
                self.__inputs,
                self.__targets,
                self.__seq_len,
                self.__logits,
                self.__decoded,
                self.__optimizer,
                self.__cost,
                self.__max_char_count,
                self.__init,
                self.__merged
            ) = self.crnn(max_image_width, batch_size)
            self.__init.run()

        with self.__session.as_default():
            #restore part of 
            #variables_to_restore = slim.get_variables_to_restore(exclude=["W","b","global_step",'learning_rate','optimizer','mini','conv2d/bias/optimizer','conv2d_1/kernel/optimizer','conv2d_1/bias/optimizer','conv2d/kernel/optimizer','conv2d_2/bias/optimizer','conv2d_2/kernel/optimizer','conv2d_3/bias/optimizer','conv2d_3/kernel/optimizer','conv2d_4/bias/optimizer','conv2d_4/kernel/optimizer','conv2d_5/bias/optimizer','conv2d_5/kernel/optimizer','conv2d_6/bias/optimizer','conv2d_6/kernel/optimizer'])
            #self.__saver = tf.train.Saver(variables_to_restore, max_to_keep=5)
            
            #restore all
            self.__saver_whole = tf.train.Saver(tf.global_variables(), max_to_keep=5)
            # Loading last save if needed
            if self.__restore:
                print('Restoring')
                ckpt = tf.train.latest_checkpoint(self.__model_path)
                if ckpt:
                    print('Checkpoint is valid')
                    self.step = int(ckpt.split('-')[1])
                    self.__saver_whole.restore(self.__session, ckpt)

        # Creating data_manager
        self.__data_manager = DataManager(iteration_count,batch_size, model_path, examples_path, max_image_width, train_test_ratio, self.__max_char_count,self.isTrain)

    def crnn(self, max_width, batch_size):
        def BidirectionnalRNN(inputs, seq_len):
            """
                Bidirectionnal LSTM Recurrent Neural Network part
            """

            with tf.variable_scope(None, default_name="bidirectional-rnn-1"):
                # Forward
                lstm_fw_cell_1 = tf.nn.rnn_cell.LSTMCell(256)
                #lstm_fw_cell_1 = rnn.BasicLSTMCell(256)
                # Backward
                lstm_bw_cell_1 = tf.nn.rnn_cell.LSTMCell(256)
                #lstm_bw_cell_1 = rnn.BasicLSTMCell(256)

                inter_output, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_1, lstm_bw_cell_1, inputs, seq_len, dtype=tf.float32)

                inter_output = tf.concat(inter_output, 2)

            with tf.variable_scope(None, default_name="bidirectional-rnn-2"):
                # Forward
                lstm_fw_cell_2 = tf.nn.rnn_cell.LSTMCell(256)
                #lstm_fw_cell_2 = rnn.BasicLSTMCell(256)
                # Backward
                lstm_bw_cell_2 = tf.nn.rnn_cell.LSTMCell(256)
                #lstm_bw_cell_2 = rnn.BasicLSTMCell(256)

                outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_2, lstm_bw_cell_2, inter_output, seq_len, dtype=tf.float32)

                outputs = tf.concat(outputs, 2)

            return outputs

        def CNN(inputs):
            """
                Convolutionnal Neural Network part
            """
            #record input image
            #inputs_ = tf.swapaxes(inputs,1,2)
            #inputs_ = tf.transpose(input, [1, 2])
            inputs_ = tf.transpose(inputs, [0, 2, 1, 3])
            tf.summary.image('raw_image',inputs_,max_outputs=10)

            #with tf.device('/gpu:0'):
                # 64 / 3 x 3 / 1 / 1
            conv1 = tf.layers.conv2d(inputs=inputs, filters = 64, kernel_size = (3, 3), padding = "same", activation=tf.nn.relu)

                # 2 x 2 / 1
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

            #88*115*16*64
            pool1_ = tf.transpose(pool1, [3, 2, 1, 0])
            #pool1_ = tf.reshape(pool1_,[-1,16,115,1])
            #pool1_ = tf.swapaxes(pool1_,1,2)
            tf.summary.image('conv1',pool1_,max_outputs=10)
            
            #with tf.device('/gpu:0'):
                # 128 / 3 x 3 / 1 / 1
            conv2 = tf.layers.conv2d(inputs=pool1, filters = 128, kernel_size = (3, 3), padding = "same", activation=tf.nn.relu)

                # 2 x 2 / 1
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
            
            pool2_ = tf.transpose(pool2, [3, 2, 1, 0])
            #pool2_ = tf.reshape(pool2_,[-1,8,57,1])
            tf.summary.image('conv2',pool2_,max_outputs=10)
            
            #with tf.device('/gpu:0'):
                # 256 / 3 x 3 / 1 / 1
            conv3 = tf.layers.conv2d(inputs=pool2, filters = 256, kernel_size = (3, 3), padding = "same", activation=tf.nn.relu)

                # Batch normalization layer
            bnorm1 = tf.layers.batch_normalization(conv3)
            
            #with tf.device('/gpu:0'):
                # 256 / 3 x 3 / 1 / 1
            conv4 = tf.layers.conv2d(inputs=bnorm1, filters = 256, kernel_size = (3, 3), padding = "same", activation=tf.nn.relu)

                # 1 x 2 / 1
            pool3 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=[1, 2], padding="same")
            
            #with tf.device('/gpu:0'):
                # 512 / 3 x 3 / 1 / 1
            conv5 = tf.layers.conv2d(inputs=pool3, filters = 512, kernel_size = (3, 3), padding = "same", activation=tf.nn.relu)

                # Batch normalization layer
            bnorm2 = tf.layers.batch_normalization(conv5)
            
            #with tf.device('/gpu:0'):
                # 512 / 3 x 3 / 1 / 1
            conv6 = tf.layers.conv2d(inputs=bnorm2, filters = 512, kernel_size = (3, 3), padding = "same", activation=tf.nn.relu)

                # 1 x 2 / 2
            pool4 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[2, 2], strides=[1, 2], padding="same")
	            
            #with tf.device('/gpu:0'):
                # 512 / 2 x 2 / 1 / 0
            conv7 = tf.layers.conv2d(inputs=pool4, filters = 512, kernel_size = (2, 2), padding = "valid", activation=tf.nn.relu)

            return conv7


        inputs = tf.placeholder(tf.float32, [batch_size, max_width, 32, 1])

        # Our target output
        targets = tf.sparse_placeholder(tf.int32, name='targets') 

        # The length of the sequence
        seq_len = tf.placeholder(tf.int32, [None], name='seq_len')
        
        cnn_output = CNN(inputs)

        reshaped_cnn_output = tf.reshape(cnn_output, [batch_size, -1, 512])

        rnn_input = tf.reshape(reshaped_cnn_output,[batch_size,-1,512,1])
        tf.summary.image('rnn_input',rnn_input,max_outputs=10)
        
        max_char_count = reshaped_cnn_output.get_shape().as_list()[1]

        crnn_model = BidirectionnalRNN(reshaped_cnn_output, seq_len)

        logits = tf.reshape(crnn_model, [-1, 512])

        W = tf.Variable(tf.truncated_normal([512, config.NUM_CLASSES], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0., shape=[config.NUM_CLASSES]), name="b")
        
        logits = tf.matmul(logits, W) + b

        logits = tf.reshape(logits, [batch_size, -1, config.NUM_CLASSES])

        # Final layer, the output of the BLSTM
        logits = tf.transpose(logits, (1, 0, 2))

        # Loss and cost calculation
        loss = tf.nn.ctc_loss(targets, logits, seq_len)

        cost = tf.reduce_mean(loss)

        # Training step
        learning_rate = tf.train.exponential_decay(1e-6,self.global_step,decay_steps=54117/self.batch_size,decay_rate=0.98,staircase=True,name="learning_rate")
        optimizer = tf.train.AdamOptimizer(learning_rate,name='optimizer').minimize(cost,global_step=self.global_step,name='mini')
        #optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
        tf.summary.scalar("learning_rate",learning_rate)

        # The decoded answer
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
        
        dense_decoded = tf.sparse_tensor_to_dense(decoded[0], default_value=-1)
        dense_targets = tf.sparse_tensor_to_dense(targets,default_value=-1)
        dense_targets = tf.cast(dense_targets,tf.int64)
        # The accurate rate,sdue to the unknown size of the tensor,we have to use the tensorflow way   
        #acc = tf.reduce_mean(tf.cast(tf.equal(dense_decoded,dense_targets),tf.int32))
        
        #tf.summary.scalar('acc_batch',acc)                    
                                
        init = tf.global_variables_initializer()
        
        #tensorboard
        tf.summary.scalar('loss_batch',cost)
        merged = tf.summary.merge_all()
        
        return inputs, targets, seq_len, logits, dense_decoded, optimizer, cost, max_char_count, init,merged

    def train(self, iteration_count):
        with self.__session.as_default():
            print('Training')
            train_iteration_card_acc = []
            train_iteration_num_acc = []
            train_batch_card_acc = []
            train_batch_num_acc = []
            test_iteration_card_acc = []
            test_iteration_num_acc = []
            counts_train = 0
            counts_test = 0
            for i in range(self.step, iteration_count + self.step):
                #iter_loss = 0
                acc_card_train_ = 0
                sum_card_train_ = 0
                acc_num_train_ = 0
                sum_num_train_ = 0
                iter_loss_train = 0
                
                acc_card_test_ = 0
                sum_card_test_ = 0
                acc_num_test_ = 0
                sum_num_test_ = 0
                iter_loss_test = 0
                
                #train
                #for k in range(1):
                    #batch_y,batch_dt,batch_x = self.__data_manager.train_batches[0]
                for batch_y, batch_dt, batch_x in self.__data_manager.train_batches:
                    counts_train = counts_train + 1
                    acc_card_train = 0
                    sum_card_train = 0
                    acc_num_train = 0
                    sum_num_train = 0
                    #for batch_y, batch_dt, batch_x in self.__data_manager.train_batches:
                    op, decoded, loss_value, train_summary = self.__session.run(
                        [self.__optimizer, self.__decoded, self.__cost, self.__merged],
                        feed_dict={
                            self.__inputs: batch_x,
                            self.__seq_len: [self.__max_char_count] * self.__data_manager.batch_size,
                            self.__targets: batch_dt
                        }
                    )
                    #print(decoded)
                    print(batch_y[0])
                    print(ground_truth_to_word(decoded[0]))
                    for j in range(self.__data_manager.batch_size):
                        true_label = batch_y[j]
                        predict_label = ground_truth_to_word(decoded[j])
                        #print(true_label)
                        #print(predict_label)
                        if true_label == predict_label:
                            acc_card_train = acc_card_train + 1
                            acc_num_train = acc_num_train + len(true_label)
                            sum_num_train = sum_num_train + len(true_label)
                        else:
                            sum_num_train = sum_num_train + len(true_label)
                            for i in range(min(len(true_label),len(predict_label))):
                                if true_label[i] == predict_label[i]:
                                    acc_num_train = acc_num_train + 1
                        sum_card_train = sum_card_train + 1
                
                    iter_loss_train += loss_value
                    acc_card_train_ += acc_card_train
                    sum_card_train_ += sum_card_train
                    acc_num_train_ += acc_num_train
                    sum_num_train_ += sum_num_train
                    
                    train_batch_card_acc.append(acc_card_train/sum_card_train)
                    train_batch_num_acc.append(acc_num_train/sum_num_train)
                    print('[{}] batch train_card_acc: {}'.format(self.step,acc_card_train/sum_card_train))
                    print('[{}] batch train_number_acc: {}'.format(self.step,acc_num_train/sum_num_train))
                    
                    self.train_writer.add_summary(train_summary,counts_train)
                    
                train_iteration_card_acc.append(acc_card_train_/sum_card_train_)
                train_iteration_num_acc.append(acc_num_train_/sum_num_train_)          
                print('[{}] iteration train_loss: {}'.format(self.step, iter_loss_train))                 
                print('[{}] iteration train_card_acc: {}'.format(self.step, acc_card_train_/sum_card_train_))
                print('[{}] iteration train_number_acc: {}'.format(self.step, acc_num_train_/sum_num_train_))
                print('\n')
                
                #test
                for batch_y, batch_dt, batch_x in self.__data_manager.test_batches:
                    counts_test = counts_test + 1
                    acc_card_test = 0
                    sum_card_test = 0
                    acc_num_test = 0
                    sum_num_test = 0      
                    #for batch_y, batch_dt, batch_x in self.__data_manager.train_batches:
                    decoded, loss_value, test_summary = self.__session.run(
                        [self.__decoded, self.__cost, self.__merged],
                        feed_dict={
                            self.__inputs: batch_x,
                            self.__seq_len: [self.__max_char_count] * self.__data_manager.batch_size,
                            self.__targets: batch_dt
                        }
                    )
                    for j in range(self.__data_manager.batch_size):
                        true_label = batch_y[j]
                        predit_label = ground_truth_to_word(decoded[j])
                        #print(true_label)
                        #print(predit_label)
                        if true_label == predit_label:
                            acc_card_test = acc_card_test + 1
                            acc_num_test = acc_num_test + len(true_label)
                            sum_num_test = sum_num_test + len(true_label)
                        else:
                            sum_num_test = sum_num_test + len(true_label)
                            for i in range(min(len(true_label),len(predict_label))):
                                if true_label[i] == predict_label[i]:
                                    acc_num_test = acc_num_test + 1
                        sum_card_test = sum_card_test + 1
                    iter_loss_test += loss_value
                    acc_card_test_ += acc_card_test
                    sum_card_test_ += sum_card_test
                    acc_num_test_ += acc_num_test
                    sum_num_test_ += sum_num_test
                    
                    self.test_writer.add_summary(test_summary,counts_test)
                    
                test_iteration_card_acc.append(acc_card_test_/sum_card_test_)
                test_iteration_num_acc.append(acc_num_test_/sum_num_test_)          
                print('[{}] iteration test_loss: {}'.format(self.step, iter_loss_test))                 
                print('[{}] iteration test_card_acc: {}'.format(self.step, acc_card_test_/sum_card_test_))
                print('[{}] iteration test_number_acc: {}'.format(self.step, acc_num_test_/sum_num_test_))
                print('\n')           
                
                if self.step % 1 == 0:
                    self.__saver_whole.save(
                        self.__session,
                        self.__save_path,
                        global_step=self.step
                    )
                                
                self.step += 1

                #iter_loss_tensor = tf.constant(iter_loss,tf.float32)
                #test_acc_tensor = tf.constant(acc_num*1.0/sum_num,tf.float32)
                #loss_sum = tf.summary.scalar("total_loss",iter_loss_tensor)
                #acc_sum = tf.summary.scalar("test_acc",test_acc_tensor)
                #merged = self.__session.run(tf.summary.merge_all())
                 
        np.savetxt("./train_batch_num_acc.txt",train_batch_num_acc)
        np.savetxt("./train_batch_card_acc.txt",train_batch_card_acc)
        np.savetxt("./train_iteration_num_acc.txt",train_iteration_num_acc)
        np.savetxt("./train_iteration_card_acc.txt",train_iteration_card_acc)
        np.savetxt("./test_iteration_num_acc.txt",test_iteration_num_acc)
        np.savetxt("./test_iteration_card_acc.txt",test_iteration_card_acc)                    
        self.train_writer.close()
        self.test_writer.close()
        return None

    def test(self):
        #f = open('../demo/result.txt','w')
        predict_result = []
        with self.__session.as_default():
            print('Testing')
            for batch_y, batch_x in self.__data_manager.test_batches:
                decoded = self.__session.run(
                    self.__decoded,
                    feed_dict={
                        self.__inputs: batch_x,
                        self.__seq_len: [self.__max_char_count] * self.__data_manager.batch_size
                    }
                )

                for i, y in enumerate(batch_y):
                    print('pic_name:')
                    print(batch_y[i])
                    name = ''.join(batch_y[i])
                    print('predict:')
                    print(ground_truth_to_word(decoded[i]))
                    ans = ''.join(ground_truth_to_word(decoded[i]))
                    str = name + ':'+ ans + '\n'
                    #f.writelines(str)
                    predict_result.append(str)
        #f.close()
        tf.reset_default_graph()
        return predict_result
