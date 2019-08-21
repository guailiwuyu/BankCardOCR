import argparse
import os
from crnn import CRNN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def parse_arguments():
    """
        Parse the command line arguments of the program.
    """

    parser = argparse.ArgumentParser(description='Train or test the CRNN model.')

    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        nargs="?",
        help="The path where the pretrained model can be found or where the model will be saved",
        default='./save/'
    )
    parser.add_argument(
        "-ex",
        "--examples_path",
        type=str,
        nargs="?",
        help="The path to the file containing the examples (training samples)",
        required=True
    )
    parser.add_argument(
        "-op",
        "--output_path",
        type=str,
        nargs="?",
        help="The path to the file output txt",
        default='../demo/test_result/result.txt'
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        nargs="?",
        help="Size of a batch",
        default=1
    )
    parser.add_argument(
        "-it",
        "--iteration_count",
        type=int,
        nargs="?",
        help="How many iteration in training",
        default=10
    )
    parser.add_argument(
        "-miw",
        "--max_image_width",
        type=int,
        nargs="?",
        help="Maximum width of an example before truncating",
        default=230
    )

    parser.add_argument(
        "-r",
        "--restore",
        action="store_true",
        help="Define if we try to load a checkpoint file from the save folder"
    )

    return parser.parse_args()

def main():
    """
        Entry point when using CRNN from the commandline
    """

    args = parse_arguments()


    crnn = None
    
    if crnn is None:
        crnn = CRNN(
            args.iteration_count,
            args.batch_size,
            args.model_path,
            args.examples_path,
            args.max_image_width,
            0,  #train/test ratio   here train rate is 0
            args.restore,
            1
        )

    predict_result = crnn.test()
    f = open(args.output_path,'w')
    for str in predict_result:
        str1 = str.split(':')[0]
        str2 = str.split(':')[1]
        str2 = str2.strip('_')
        f.writelines(str1+':'+str2)
    f.close()

if __name__ == '__main__':
    main()
