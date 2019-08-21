# -*- coding: cp936 -*-

import  re

#s="_020a_0@1.png"

def getIndex(s):
    result = ""
    max_index = len(s)-1
    flag = 0
    for index,value in enumerate(s):
        if(s[index] == '@'):
            flag = 1
            continue
        if(flag == 1):
            if(s[index] != '.'):
                result+=s[index]
        if(s[index] == '.'):
            break
    return result

