import os
from sklearn.model_selection import train_test_split
from src.model import tokenizer, WineReviewT5
from src.dataset import WineReviewDataset

def train():
    data_dir = os.path.join(os.getcwd(), 'data')
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    
    log_dir = os.path.join(data_dir, 'experiments', 't5', 'logs')
    save_path = os.path.join(data_dir, 'experiments', 't5', 'models')
    cache_path_train = os.path.join(data_dir, 'cache', 't5.train')
    cache_path_test = os.path.join(data_dir, 'cache', 't5.test')

    dataset = WineReviewDataset()
    dataset.shuffle()
    dataset = dataset.transform("review")

    tokenizer = tokenizer()