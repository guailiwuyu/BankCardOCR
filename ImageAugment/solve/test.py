import os
def rename(path):
    #去路径 返回文件名
    file_name = os.path.basename(path)
    filelist = os.listdir(path) #获取文件路径
    total_num = len(filelist) #获取文件长度（个数）
    for item in filelist:
        if item.endswith('.jpg'):  #初始的图片的格式为jpg格式的（或者源文件是png格式及其他格式，后面的转换格式就可以调整为自己需要的格式即可）
            src = os.path.join(os.path.abspath(path), item)
            dst = os.path.join(item.replace('~','#'))
            try:
                os.rename(src, dst)
                print ('converting %s to %s ...' % (src, dst))
                i = i + 1
            except:
                continue
    #print ('total %d to rename & converted %d jpgs' % (total_num, i))
#rename('cardnumber')