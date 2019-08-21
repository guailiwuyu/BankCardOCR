from __future__ import print_function

import cv2
import glob
import os
import shutil
import sys
import time
import argparse
import numpy as np
import tensorflow as tf

sys.path.append(os.getcwd())
from crop2 import rotate
from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg, cfg_from_file
from lib.fast_rcnn.test import test_ctpn
from lib.utils.timer import Timer
from lib.text_connector.detectors import TextDetector
from lib.text_connector.text_connect_cfg import Config as TextLineCfg


def resize_im(im, scale, max_scale=None):
    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])
    return cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR), f
    
#文本识别并保存中间结果
def draw_boxes(img,boxes,image_name, out_path,scale):
    #生成的图片的名称
    base_name = image_name.split('\\')[-1] 

    flag = False
    #令最长的框为第一个概率大于0.8的框
    for box in boxes:
        if box[8] >= 0.8:           
            maxBox = box
            flag = True
    #找出最长的框
    for box in boxes:
        if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
           continue
        if((box[2]-box[0])>=(maxBox[2]-maxBox[0]) and box[8]>=0.8):
            maxBox = box
    #如果框存在则对图片进行画框
    if flag==True:
        if maxBox[8] >= 0.8:
           color = (0, 255, 0)
        #box[]长度为9,box[8]代表框的概率
        #从上到下，从左到右，矩形框的四个坐标为(box[0],box[1]),(box[2],box[3]),(box[4],box[5]),(box[6],box[7])
        cv2.line(img, (int(maxBox[0]), int(maxBox[1])), (int(maxBox[2]), int(maxBox[3])), color, 2)
        cv2.line(img, (int(maxBox[0]), int(maxBox[1])), (int(maxBox[4]), int(maxBox[5])), color, 2)
        cv2.line(img, (int(maxBox[6]), int(maxBox[7])), (int(maxBox[2]), int(maxBox[3])), color, 2)
        cv2.line(img, (int(maxBox[4]), int(maxBox[5])), (int(maxBox[6]), int(maxBox[7])), color, 2)
        #print(int(maxBox[0])/scale,int(maxBox[1])/scale,int(maxBox[2])/scale,int(maxBox[3])/scale,int(maxBox[4])/scale,int(maxBox[5])/scale,int(maxBox[6])/scale,int(maxBox[7])/scale)
    img = cv2.resize(img, None, None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(out_path+"\\"+base_name, img)
    
#找出最长的框并画出来
def draw_boxes2(img,boxes,image_name ,out_path,scale):
    flag = False
    #令最长的框为第一个概率大于0.8的框
    for box in boxes:
        if box[8] >= 0.8:           
            maxBox = box
            flag = True
    #找出最长的框
    for box in boxes:
        if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
           continue
        if((box[2]-box[0])>=(maxBox[2]-maxBox[0]) and box[8]>=0.8):
            maxBox = box
    #如果框都小于0.8 
    if flag == False:
        base_name = image_name.split('\\')[-1]
        cv2.imwrite(out_path+"\\"+base_name, img)
    #如果框存在则对图片进行裁剪
    if flag==True:
        if(maxBox[8]>=0.8):
            base_name = image_name.split('\\')[-1]
            rotate(img,[int(maxBox[0]),int(maxBox[1])],[int(maxBox[4]),int(maxBox[5])],[int(maxBox[6]),int(maxBox[7])],[int(maxBox[2]),int(maxBox[3])],out_path+"\\"+base_name)
        #     #找到maxY，minY，maxX，minX
        #     if(maxBox[3]<maxBox[1]):
        #         minY = maxBox[3]
        #     else:
        #         minY = maxBox[1]
        #     if(maxBox[5]>maxBox[7]):
        #         maxY = maxBox[5]
        #     else:
        #         maxY = maxBox[7]
        #     if(maxBox[2]>maxBox[6]):
        #         maxX = maxBox[2]
        #     else:
        #         maxX = maxBox[6]
        #     if(maxBox[0]<maxBox[4]):
        #         minX = maxBox[0]
        #     else:
        #         minX = maxBox[4]
        #     if minY<0:
        #         minY=0
        #     if minX<0:
        #         minX=0
        #     img = img[int(minY):int(maxY),int(minX):int(maxX)]
        #     img = cv2.resize(img, None, None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_LINEAR)       
        #     #cv2.imshow('image',img)
        #     #cv2.waitKey(0)
        #     base_name = image_name.split('\\')[-1]
        #     cv2.imwrite(out_path+"\\"+base_name, img)
        #     #print(out_path+"/"+base_name)
        else: 
            print("There is no box")
        
#画出所有的小文本框
def draw_boxes3(img, boxes,image_name,  scale):
    base_name = image_name.split('\\')[-1]
    color = (0, 255, 0)
    with open('data/results/' + 'res_{}.txt'.format(base_name.split('.')[0]), 'w') as f:
        for box in boxes:
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[1])), color, 2)
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[0]), int(box[3])), color, 2)
            cv2.line(img, (int(box[2]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[0]), int(box[3])), (int(box[2]), int(box[3])), color, 2)

    img = cv2.resize(img, None, None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join("data/results", base_name), img)

def ctpn(sess, net, image_name,save_path1,save_path2):
    timer = Timer()
    timer.tic()

    #读取图片
    img = cv2.imread(image_name)
    img, scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE) 
    #灰度化处理
    #img2 = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    #img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2RGB)
#     base_name = im_name.split('\\')[-1]
#     cv2.imwrite(os.path.join("data/results2", base_name), img2)
       
    scores, boxes = test_ctpn(sess, net, img)

    #后处理过程，detect包含过滤和合并
    textdetector = TextDetector()
    boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
    draw_boxes2(img, boxes,image_name, save_path2,scale)
    draw_boxes(img, boxes,image_name, save_path1,scale)
    
    #后处理过程，detect2只过滤小文本框
#     textdetector = TextDetector()
#     boxes = textdetector.detect2(boxes, scores[:, np.newaxis], img.shape[:2])
#     draw_boxes3(img, boxes,image_name, scale)
    
    timer.toc()
    print(('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0]))


parser = argparse.ArgumentParser(description="CTPN test single image test procedure.")
parser.add_argument("input_image", type=str,
                    help="The path of the input image.",
                    nargs="?",
                    default='data/demo',
                    )
parser.add_argument("save_path1", type=str,
                    help="The path of the output image.",
                    nargs="?",
                    default='data/results',
                    )
parser.add_argument("save_path2", type=str,
                    help="The path of the output image.",
                    nargs="?",
                    default='data/results2',
                    )
args = parser.parse_args() 
if __name__ == '__main__':
    
    start = time.time()
    
    #输入图片的路径
    im_path = args.input_image
    #输出画四边形图片的路径
    save_path1 = args.save_path1
    #输出截取图片的路径
    save_path2 = args.save_path2

    cfg_from_file('ctpn/text.yml')

    # init session
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    # load network
    net = get_network("VGGnet_test")
    # load model
    print(('Loading network {:s}... '.format("VGGnet_test")), end=' ')
    saver = tf.train.Saver()

    try:
        ckpt = tf.train.get_checkpoint_state(cfg.TEST.checkpoints_path)
        print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('done')
    except:
        raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)

    # im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    # for i in range(2):
    #     _, _ = test_ctpn(sess, net, im)

    im_names = glob.glob(os.path.join(im_path, '*.png')) + \
               glob.glob(os.path.join(im_path, '*.jpg')) + \
               glob.glob(os.path.join(im_path, '*.jpeg'))
               
    
    cost_time = (time.time() - start)
    print("cost time: {:.2f}s".format(cost_time))
    
    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(('Demo for {:s}'.format(im_name)))
        start = time.time()
        ctpn(sess, net, im_name,save_path1,save_path2)
        cost_time = (time.time() - start)
        print("cost time: {:.2f}s".format(cost_time))
    #清空默认图
    tf.reset_default_graph()
