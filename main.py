from statistics import mode
from readprocessedcsv import readProcessedCsv
from dataset import Dataset, TestDataset
from model import GCN
from train import train
from test import test

import torch

def main():
    
    train_source_tweet_map = readProcessedCsv(all_csv_file_path="./processed_data/2019+pheme_task_b_training/raw/")
    config_path = "./roberta-base/config.json"
    checkpoint = "roberta-base"
    model_path = "./roberta-base/pytorch_model.bin"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = Dataset(root="./processed_data/2019+pheme_task_b_training/processed/", belong_to_which_source_tweet_map=train_source_tweet_map, config_path=config_path, model_path=model_path, checkpoint=checkpoint, device=device)
    test_source_tweet_map = readProcessedCsv(all_csv_file_path="./processed_data/2017taskb_testing/raw/")
    test_dataset = TestDataset(root="./processed_data/2017taskb_testing/processed/", belong_to_which_source_tweet_map=test_source_tweet_map, config_path=config_path, model_path=model_path, checkpoint=checkpoint, device=device)
    model = GCN(16, 4, 12, 3).to(device)
    train(train_dataset, model, device)
    test(test_dataset, model, device)

if __name__ == '__main__':
    main()
    