from PIL import ImageDraw
from PIL import Image
import random

def getRandomColor():
    '''获取一个随机颜色(r,g,b)格式的'''
    c1 = random.randint(0, 255)
    c2 = random.randint(0, 255)
    c3 = random.randint(0, 255)
    return (c1, c2, c3)
def getRandomStr():
    '''获取一个随机字符串，每个字符的颜色也是随机的'''
    random_num = str(random.randint(0, 9))
    random_low_alpha = chr(random.randint(97, 122))
    random_upper_alpha = chr(random.randint(65, 90))
    random_char = random.choice([random_num, random_low_alpha, random_upper_alpha])
    return random_char

def add_line_point(image):
    # 获取一个Image对象，参数分别是RGB模式。宽150，高30，随机颜色

    # 获取一个画笔对象，将图片对象传过去
    draw = ImageDraw.Draw(image)

    # 噪点噪线
    width = image.width
    height = image.height
    # 划线
    for i in range(3):
        x1 = random.randint(0, width)
        x2 = random.randint(0, width)
        y1 = random.randint(0, height)
        y2 = random.randint(0, height)
        draw.line((x1, y1, x2, y2), fill=getRandomColor())

    # 画点
    for i in range(10):
        draw.point([random.randint(0, width), random.randint(0, height)], fill=getRandomColor())
        x = random.randint(0, width)
        y = random.randint(0, height)
        draw.arc((x, y, x + 4, y + 4), 0, 90, fill=getRandomColor())
    return image


def addLineAll(image,dest):
    L = [image,image,image,image,image]
    for i in range(0,3):
        x = L[i]
        #x.show()
        ans_img = add_line_point(x)
        ans_dir = dest + "Line" + str(i) + ".png"
        ans_img.save(ans_dir)
