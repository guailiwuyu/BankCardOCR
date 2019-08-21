# -*- coding: utf-8 -*-
from flask import Flask,render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import time
import base64
from datetime import  timedelta
import sys
sys.path.append('../card_number_detection/ctpn/')
sys.path.append('../card_number_recognition/')
sys.path.append('../ImageAugment/solve/')
os.chdir('../card_number_detection')
import demo_single
from demo_single import detection
os.chdir('../card_number_recognition')
import recognition
from recognition import recognition2
from main_solve import augment2
os.chdir('../GUI')

#设置允许的文件格式
ALLOWED_EXTENSIONS = set(['png','PNG','jpg','JPG','jpeg','JPEG','bmp','BMP'])
def allowed_file(filename):
    return '.'in filename and filename.rsplit('.',1)[1] in ALLOWED_EXTENSIONS

def makedir(filepath):    
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    

app = Flask(__name__) # 实例化一个flask 对象
#设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)

#添加路由
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/augment')
def augment():
    return render_template('augment.html')

@app.route('/recognition')
def recognition():
    return render_template('recognition.html')

@app.route('/imageAugment',methods=['GET','POST'])
def imageAugment():
    if request.method=='POST':
        f = request.files['file']
        
        if not(f and allowed_file(f.filename)):
            return "error"
        
        #basepath = os.path.dirname(__file__)      #当前文件所在路径
        timeStamp = str(int(time.time()))          #时间戳
         
        orign_path = os.path.join('static/upload/'+timeStamp+'/orign/')    #原图片的保存路径
        augment_path = os.path.join('static/upload/'+timeStamp+'/augment/')#增强后的图片的保存路径
        upload_path = orign_path+secure_filename(f.filename)                        
        makedir(orign_path)
        makedir(augment_path)
        #保存并更改名字
        f.save(upload_path)
        os.rename(upload_path,orign_path+timeStamp+'.jpg')
        #执行增强脚本
        #cmd = 'python ../ImageAugment/solve/main_solve.py '+orign_path+' '+augment_path
        #os.system(cmd)
        augment2(orign_path,augment_path)
        return '88|'+timeStamp

#通过file传输图片
@app.route('/processingSingle',methods=['GET','POST'])
def processingSingle():
    start = time.time()
    if request.method=='POST':
        f = request.files['file']
        
        if not(f and allowed_file(f.filename)):
            return "error1"

        #basepath = os.path.dirname(__file__)  #当前文件所在路径
        timeStamp = str(int(time.time()))          #时间戳 
        orign_path = os.path.join('static/upload/'+timeStamp+'/orign/')          #原图片的保存路径
        detection1_path = os.path.join('static/upload/'+timeStamp+'/detection1/')#检测中间结果1
        detection2_path = os.path.join('static/upload/'+timeStamp+'/detection2/')#检测中间结果2
        makedir(orign_path)
        makedir(detection1_path)
        makedir(detection2_path)
        
        #保存并更改名字
        upload_path = orign_path+secure_filename(f.filename)
        f.save(upload_path)
        os.rename(upload_path,orign_path+timeStamp+'.jpg')
        
        cost_time = (time.time() - start)
        print("上传图片花费时间cost time: {:.2f}s".format(cost_time))
        
        #执行定位脚本
        load_path = '../GUI/'+orign_path+timeStamp+'.jpg'
        output_path1 = '../GUI/'+detection1_path
        output_path2 = '../GUI/'+detection2_path
        #切换当前工作环境并执行cmd命令
        os.chdir('../card_number_detection')
        #判断路径是否存在
        if not (os.path.exists(load_path) and os.path.exists(output_path1) and os.path.exists(output_path2)):
            return "error2"
        #执行定位函数
        detection(load_path,output_path1,output_path2)        
        cost_time = (time.time() - start)
        print("卡号定位花费时间cost time: {:.2f}s".format(cost_time))
        
        #切换当前工作环境并执行cmd命令
        predictPath = '../GUI/static/upload/'+timeStamp+'/'+timeStamp+'.txt';     
        os.chdir('../card_number_recognition')
        #判断路径是否存在
        if not (os.path.exists(output_path2)):
            return "error3"
        #执行识别脚本
        recognition2(output_path2,predictPath)
        os.chdir('../GUI')
        #读取识别结果
        with open(predictPath, 'r') as file_to_read:
            line = file_to_read.readline() # 整行读取数据
        line = line.split(':')[1]
        cost_time = (time.time() - start)
        print("卡号识别花费时间cost time: {:.2f}s".format(cost_time))    
        return timeStamp+'|'+line.replace('_',' ')

#通过base64传输文件
@app.route('/processingSingle2',methods=['GET','POST'])
def processingSingle2():
    start = time.time()
    #传输base64的json文件
    data = request.json['image']

    timeStamp = str(int(time.time()))          #时间戳 
    orign_path = os.path.join('static/upload/'+timeStamp+'/orign/')          #原图片的保存路径
    detection1_path = os.path.join('static/upload/'+timeStamp+'/detection1/')#检测中间结果1
    detection2_path = os.path.join('static/upload/'+timeStamp+'/detection2/')#检测中间结果2
    makedir(orign_path)
    makedir(detection1_path)
    makedir(detection2_path)
    #图片的保存路径
    upload_path = orign_path+timeStamp+".jpg"
    #保存图片
    with open(upload_path,'wb') as fdecode:
        decode_base64 = base64.b64decode(data)
        fdecode.write(decode_base64)

    cost_time = (time.time() - start)
    print("上传图片花费时间cost time: {:.2f}s".format(cost_time))
    
    #执行定位脚本
    load_path = '../GUI/'+orign_path+timeStamp+'.jpg'
    output_path1 = '../GUI/'+detection1_path
    output_path2 = '../GUI/'+detection2_path
    #切换当前工作环境并执行cmd命令
    os.chdir('../card_number_detection')
    #判断路径是否存在
    if not (os.path.exists(load_path) and os.path.exists(output_path1) and os.path.exists(output_path2)):
        return "error2"
    #执行定位函数
    detection(load_path,output_path1,output_path2)         
    cost_time = (time.time() - start)
    print("卡号定位花费时间cost time: {:.2f}s".format(cost_time))
    
    #切换当前工作环境并执行cmd命令
    predictPath = '../GUI/static/upload/'+timeStamp+'/'+timeStamp+'.txt';     
    os.chdir('../card_number_recognition')
    #判断路径是否存在
    if not (os.path.exists(output_path2)):
        return "error3"
    #执行识别脚本
    recognition2(output_path2,predictPath)
    os.chdir('../GUI')
    #读取识别结果
    with open(predictPath, 'r') as file_to_read:
        line = file_to_read.readline() # 整行读取数据
    line = line.split(':')[1]
    line = line.replace('_',' ')
    #读取定位图片并转为base64格式
    f = open(output_path1+timeStamp+'.jpg','rb')
    base64_str = str(base64.b64encode(f.read()),'utf-8')
    #卡号识别花费时间
    cost_time = (time.time() - start)
    print("卡号识别花费时间cost time: {:.2f}s".format(cost_time)) 

    #返回结果
    return jsonify({"base64_str":base64_str,"result":line})

if __name__ == '__main__':
    app.run(debug = True)
# if __name__ == '__main__':
#     from werkzeug.contrib.fixers import ProxyFix
#     app.wsgi_app = ProxyFix(app.wsgi_app)
#     app.run()