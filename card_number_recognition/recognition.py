import argparse
import os
from crnn import CRNN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def recognition2(examples_path,output_path):
    """
        Entry point when using CRNN from the commandline
    """
    crnn = None
    if crnn is None:
        crnn = CRNN(
            10,
            1,
            "./save/",
            examples_path,
            230,
            0,  #train/test ratio   here train rate is 0
            True,
            1
        )
        
    predict_result = crnn.test()
    f = open(output_path,'w')
    for str in predict_result:
        str1 = str.split(':')[0]
        str2 = str.split(':')[1]
        str2 = str2.strip('_')
        f.writelines(str1+':'+str2)
    f.close()
