import numpy as np
import tensorflow as tf

from PIL import Image
from scipy.misc import imread, imresize, imsave

import config

def parse_exmp(serialized_example):
    features = tf.parse_single_example(serialized_example,
    features={
            'label': tf.FixedLenFeature([],tf.string),
            'img_raw': tf.FixedLenFeature([],tf.string),
            'row': tf.FixedLenFeature([],tf.int64),
            'col': tf.FixedLenFeature([],tf.int64)
           })
    image = tf.decode_raw(features['img_raw'],tf.uint8)
    img_label = features['label']
    row = tf.cast(features['row'],tf.int64)
    col = tf.cast(features['col'],tf.int64)
    
    return image,img_label,row,col

def sparse_tuple_from(sequences, dtype=np.int32):
    """
        Inspired (copied) from https://github.com/igormq/ctc_tensorflow_example/blob/master/utils.py
    """
    indices = []
    values = []

    #将批量的银行卡标签打包成存有其下标，数据值和大小的三个数组
    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), [i for i in range(len(seq))]))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

    return indices, values, shape

def resize_image(image, input_width):
    """
        Resize an image to the "good" input size
    """
    #print(image)
    im_arr = imread(image, mode='L')
    #print(im_arr)
    r, c = np.shape(im_arr)
    #print(np.shape(im_arr))
    if c > input_width:
        c = input_width
        ratio = float(input_width) / c
        final_arr = imresize(im_arr, (int(32 * ratio), input_width))
        #print("big")
    else:
        #如果训练图片的宽度是比resize的宽度小的话，后面全部填充为0，训练时                                      相当于把全0的部分训练成了空格
        final_arr = np.zeros((32, input_width))  
        ratio = 32.0 / r
        length = int(c * ratio)
        im_arr_resized = imresize(im_arr, (32, length))
        if length>input_width:
            final_arr = imresize(im_arr_resized,(32,input_width))
            #print("small")
        else:
            final_arr[:, 0:length] = im_arr_resized[:, 0:length]
    return final_arr, c

def resize_train_image(im_arr, input_width):
    """
        Resize an image to the "good" input size
    """
    #print("resize")
    img = Image.fromarray(im_arr)
    img = img.convert('L')
    #img.save("./1.jpg")
    img = img.resize((230,32),Image.BILINEAR)   
    final_arr = np.asarray(img)
    c = input_width
    #r, c = np.shape(im_arr)
    #print(np.shape(im_arr))
    #if c > input_width:
        #c = input_width
        #ratio = float(input_width) / c
        #final_arr = imresize(im_arr, (int(32 * ratio), input_width))
        #print("big")
    #else:
        #如果训练图片的宽度是比resize的宽度小的话，后面全部填充为0，训练时                                      相当于把全0的部分训练成了空格
        #final_arr = np.zeros((32, input_width))  
        #ratio = 32.0 / r
        #length = int(c * ratio)
        #im_arr_resized = imresize(im_arr, (32, length))
        #if length>input_width:
        #    final_arr = imresize(im_arr_resized,(32,input_width))
            #print("small")
        #else:
         #   final_arr[:, 0:length] = im_arr_resized[:, 0:length]
    return final_arr, c

def label_to_array(label):
    try:
        #print(label)
        return [config.CHAR_VECTOR.index(x) for x in label]
    except Exception as ex:
        print(label)
        raise ex

def ground_truth_to_word(ground_truth):
    """
        Return the word string based on the input ground_truth
    """

    try:
        return ''.join([config.CHAR_VECTOR[i] for i in ground_truth if i != -1])
    except Exception as ex:
        print(ground_truth)
        print(ex)
        input()

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]
