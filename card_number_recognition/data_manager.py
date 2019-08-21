import re
import os
import numpy as np
import config
import tensorflow as tf
from PIL import Image

from utils import parse_exmp,sparse_tuple_from, resize_image, label_to_array,resize_train_image

from scipy.misc import imsave

class DataManager(object):
    def __init__(self,iteration_count, batch_size, model_path, examples_path, max_image_width, train_test_ratio, max_char_count,isTrain):
        
        if train_test_ratio > 1.0 or train_test_ratio < 0:
            raise Exception('Incoherent ratio!')

        print(train_test_ratio)
        
        #tensorboard
        self.iteration_count = iteration_count
        self.train_test_ratio = train_test_ratio
        self.max_image_width = max_image_width
        self.batch_size = batch_size
        self.model_path = model_path
        self.current_train_offset = 0
        self.examples_path = examples_path
        self.max_char_count = max_char_count
        self.data, self.data_len = (self.__load_train_data() if isTrain == 0 else self.__load_data())
        self.test_offset = int(train_test_ratio * self.data_len)
        self.current_test_offset = self.test_offset
        self.train_batches = self.__generate_all_train_batches()
        #for batch test
        self.test_batches = self.__generate_all_test_batches()

    def __load_data(self):
        """
            Load all the images in the folder
        """

        print('Loading data')

        examples = []

        count = 0
        skipped = 0
        for f in os.listdir(self.examples_path):
            #消除大于指定长度序列的图片，并且在jupyter环境下保证忽略.ipynb_checkpoint文件
            #if len(f.split('~')[0]) > self.max_char_count or f.startswith('.'):
            #    continue
            arr, initial_len = resize_image(
                os.path.join(self.examples_path, f),
                self.max_image_width
            )
            
            examples.append(
                (
                    arr,
                    f.split('.')[0]
                    #label_to_array(f.split('~')[0])
                )
            )
            #imsave('blah.png', arr)
            count += 1
        #print(examples)
        return examples, len(examples)

    def __load_tfRecord(self):
        """
        load tfRecord
        """
        print("loading train data")

        files = os.path.join(self.examples_path,'*.tfrecords')
        
        dataset = tf.data.TFRecordDataset(['../data/train/trainCard1.tfrecords',
                               '../data/train/trainCard2.tfrecords',
                               '../data/train/trainCard3.tfrecords',
                              '../data/train/trainCard4.tfrecords',
                              '../data/train/trainCard5.tfrecords',
                               '../data/train/trainCard6.tfrecords',
                               '../data/train/trainCard7.tfrecords',
                               '../data/train/generateCard1.tfrecords',
                               '../data/train/generateCard2.tfrecords'])
        dataset = dataset.map(parse_exmp)
        #确保充分打乱
        print("batch size {},iter size {}".format(self.batch_size,self.iteration_count))
        #不定长取batch很麻烦，pad有问题，我自己取batch,在后面的函数中
        dataset = dataset.batch(1).repeat(self.iteration_count).shuffle(100000)  
        #变长需要pad
        #dataset = dataset.padded_batch(self.batch_size,padded_shapes=([None,None],[None],[None],[None]))
        #一个一个的迭代
        iterator = dataset.make_one_shot_iterator()  
        
        return iterator
    
    def __generate_tfRecord_batch(self):
        """
        load one batch tfRecord
        """
            
        examples = []        
        images,labels,heights,widths = self.iterator.get_next()
        
        with tf.Session() as sess:
            image_batch,label_batch,height_batch,width_batch = sess.run([images,labels,heights,widths])        
            for i in range(self.batch_size):
                #不定长的tensorflow的batch太麻烦了，不如我按1取，然后自己batch  

                image1 = image_batch[0]
                label1 = label_batch[0]
                height = height_batch[0]
                width = width_batch[0]

                #parse to data type
                label1 = bytes.decode(label1)

                #print(label1)

                image1 = np.reshape(np.array(image1),(height,width))
                arr, initial_len = resize_train_image(image1,self.max_image_width)               

                #保存到tensorboard里观察
                #r,c = np.shape(arr)
                #new_img = tf.reshape(arr,(r,c))
                # pic_num = pic_num + 1
                #pics.append(new_img)

                examples.append(
                           (
                            arr,
                            label1,
                            label_to_array(label1)
                           ) )    
        #print(len(examples))
        return examples
                
    def preprocess_one_train_batch(self):
        """
        preprocess
        """
        
        batch_tf_data = self.__generate_tfRecord_batch()
        raw_batch_x, raw_batch_y, raw_batch_la = zip(*batch_tf_data)

        batch_y = np.reshape(
            np.array(raw_batch_y),
            (-1)
         )

        batch_dt = sparse_tuple_from(
                #np.reshape(
                 np.array(raw_batch_la),
                #    (-1)
               # )
            )

        raw_batch_x = np.swapaxes(raw_batch_x, 1, 2)

        batch_x = np.reshape(
               np.array(raw_batch_x),
               (len(raw_batch_x), self.max_image_width, 32, 1)
         )

        #train_batches.append((batch_y, batch_dt, batch_x))
        print("get one batch")
        return batch_y,batch_dt,batch_x        
    
    def __load_train_data(self):
        """
        load all train data
        """
        print("loading train data")
        examples = []
        
        filename = './data/train/*.tfrecords'
        files = tf.train.match_filenames_once(filename)
        filename_queue = tf.train.string_input_producer(files,shuffle=True,num_epochs=1)
        reader = tf.TFRecordReader()
        _,serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example,
                                           features={
                                                'label': tf.FixedLenFeature([],tf.string),
                                                'img_raw': tf.FixedLenFeature([],tf.string),
                                                'row': tf.FixedLenFeature([],tf.int64),
                                                'col': tf.FixedLenFeature([],tf.int64)
                                           })
        image = tf.decode_raw(features['img_raw'],tf.uint8)
        img_label = features['label']  #tf中的字符串是以二进制存储的,用bytes.decode解码一下就好
        row = tf.cast(features['row'],tf.int64)
        col = tf.cast(features['col'],tf.int64)
        with tf.Session() as sess:
            #函数内部可能定义了局部变量，还有自己定义的全局变量，在run之前一定要把所有变量初始化
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            images = []
            i = 0
            try:
                while True:
                    i = i + 1
                    image1,label1,height,width = sess.run([image,img_label,row,col])

                    label1 = bytes.decode(label1) #对二进制存储进行转换
                    image1 = np.reshape(np.array(image1),(height,width))
                    arr, initial_len = resize_train_image(image1,self.max_image_width)
                    strs = label1.split('_')
                    label_without_ = ''.join(strs[i] for i in range(len(strs)))
                    #print(label_without_)
                    #保存到tensorboard里观察                
                    examples.append(
                        (
                            arr,
                            label_without_,
                            label_to_array(label_without_)
                        )
                    )            
                    
            except tf.errors.OutOfRangeError:
                print("done!")
        print(i)
        coord.request_stop()
        coord.join(threads)
            
        return examples,len(examples)        
    
    def __generate_all_train_batches(self):
        train_batches = []
        while not self.current_train_offset + self.batch_size > self.test_offset:
            old_offset = self.current_train_offset

            new_offset = self.current_train_offset + self.batch_size

            self.current_train_offset = new_offset

            raw_batch_x, raw_batch_y, raw_batch_la = zip(*self.data[old_offset:new_offset])

            batch_y = np.reshape(
                np.array(raw_batch_y),
                (-1)
            )

            batch_dt = sparse_tuple_from(
                #np.reshape(
                 np.array(raw_batch_la),
                #    (-1)
               # )
            )

            raw_batch_x = np.swapaxes(raw_batch_x, 1, 2)

            batch_x = np.reshape(
                np.array(raw_batch_x),
                (len(raw_batch_x), self.max_image_width, 32, 1)
            )

            train_batches.append((batch_y, batch_dt, batch_x))
        return train_batches

    def __generate_all_test_batches(self):
        test_batches = []
        while not self.current_test_offset + self.batch_size > self.data_len:
            old_offset = self.current_test_offset

            new_offset = self.current_test_offset + self.batch_size

            self.current_test_offset = new_offset
            
            #列表中的对应元素打包成元组，所有元祖构成一个列表
            raw_batch_x, raw_batch_y = zip(*self.data[old_offset:new_offset])

            batch_y = np.reshape(
                np.array(raw_batch_y),
                (-1)
            )

            #print(batch_dt)
            raw_batch_x = np.swapaxes(raw_batch_x, 1, 2)

            batch_x = np.reshape(
                np.array(raw_batch_x),
                (len(raw_batch_x), self.max_image_width, 32, 1)
            )
            
            test_batches.append((batch_y, batch_x))
        return test_batches
