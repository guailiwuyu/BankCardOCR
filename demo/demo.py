import os
import time

from datetime import  timedelta

if __name__ == '__main__':
    start = time.time()
    print('加载定位模型参数.........')
    #切换当前工作环境并执行cmd命令
    os.chdir('../card_number_detection')
    cmd1 = 'python ./ctpn/demo.py ../demo/test_images ../demo/test_result ../demo/test_result2'
    os.system(cmd1)
    print('定位完成')

    print('加载识别模型参数.........')
    #切换当前工作环境并执行cmd命令
    os.chdir('../card_number_recognition')
    cmd2 = 'python test.py -r -ex ../demo/test_result2 -op ../demo/test_result/result.txt'
    os.system(cmd2)
    print('识别完成')

    #切换当前工作环境
    os.chdir('../demo')