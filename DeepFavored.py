import os, sys, argparse
from train import TrainWithArgs
from identify import IdentifyWithArgs

def full_parser():
    parser=argparse.ArgumentParser(description="DeepFavored: Deep learning networks identifying favored(adaptive) mutations.")
    subparsers = parser.add_subparsers(help="sub-commands")
    train_parser = subparsers.add_parser('train', help="train DeepFavored")
    train_parser.add_argument('--config', action='store', type=str, help="config file containing the hyper-params for training deepfavored")
    train_parser.add_argument('--trainData', action='store', type=str, help="directory containing training data")
    train_parser.add_argument('--modelDir', action='store', type=str, help="directory used to contain the trained model")
    identify_parser = subparsers.add_parser('identify', help="identify the SNPs carrying favored mutation")
    identify_parser.add_argument('--modelDir', action='store', type=str, help="directory containing trained model")
    identify_parser.add_argument('--input', action='store', type=str, help="directory or a file, containing the data to be identified")
    identify_parser.add_argument('--outDir', action='store', type=str, help="directory used to contain the identified data")
    return parser

if __name__ == '__main__':
    runparser = full_parser()
    args = runparser.parse_args()

    # if called with no arguments, print help
    if len(sys.argv)==1:
        runparser.parse_args(['--help'])
    
    if sys.argv[1] == 'train':
        function_name = 'TrainWithArgs(args)'
    if sys.argv[1] == 'identify':
        function_name = 'IdentifyWithArgs(args)'
    
    eval(function_name)

