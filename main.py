import argparse
from src.train import train, test

parser = argparse.ArgumentParser(description='sommelier transformer',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-M', '--mode', default='test', choices=['train', 'test', 'debug'], dest='mode', help='train or test.')
args = parser.parse_args()

if __name__=='__main__':
    mode = args.mode
    
    if mode=='train':
        train()
    elif mode=='test':
        test()
    elif mode=='debug':
        train(shards=True)