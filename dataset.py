import os.path as osp
from tqdm import tqdm
import torch
from torch_geometric.data import Data, Dataset
from transformers import AutoConfig, AutoTokenizer, AutoModel

class Dataset(Dataset):

    def __init__(self, root, belong_to_which_source_tweet_map=None, config_path=None, model_path=None, checkpoint=None, device=None, transform=None, pre_transform=None, pre_filter=None):
        
        self.belong_to_which_source_tweet_map = belong_to_which_source_tweet_map
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.config = AutoConfig.from_pretrained(config_path)
        self.model = AutoModel.from_pretrained(model_path, config=self.config).to(device).eval()
        self.idx = 0
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return ["./processed_data/2019+pheme_task_b_training/raw/features_extraction_6.csv", "./processed_data/2019+pheme_task_b_training/raw/pheme_rumours_features_extraction_with_label_a.csv"]
    
    @property
    def processed_file_names(self):
        # return os.listdir("./processed_data/2019+pheme_task_b_training/processed/")
        return []

    def _download(self):
        pass

    def process(self):
        for key in tqdm(self.belong_to_which_source_tweet_map):
            temp_id_list = []
            temp_text_list = []
            
            x = None
            edge_index = None
            y_a = None
            y_b = torch.tensor([ self.belong_to_which_source_tweet_map[key]["label"] ], dtype=torch.long)
            
            for index, item_dict in enumerate(self.belong_to_which_source_tweet_map[key]["item_dict_list"]):
                if(item_dict["twitter_id"] == item_dict["belong_to_which_source_tweet"]): 
                    temp_id_list.append(item_dict["twitter_id"])
                    temp_text_list.append(item_dict["text"])

                    with torch.no_grad():
                        tokenized_input = self.tokenizer(item_dict["text"], item_dict["text"], return_token_type_ids=True, 
                                                    padding='max_length', max_length=128, truncation=True, return_tensors="pt").to(self.device)
                        temp_x = self.model(**tokenized_input)[0][:, 0, :]
                    x = temp_x.to(torch.device("cpu")) 
                    del tokenized_input
                    del temp_x
                    
                    y_a = torch.tensor([item_dict["label_a"]], dtype=torch.long)

                else:
                    temp_id_list.append(item_dict["twitter_id"])
                    temp_text_list.append(item_dict["text"])

                    try: 
                        with torch.no_grad():
                            tokenized_input = self.tokenizer(temp_text_list[0], item_dict["text"], return_token_type_ids=True, 
                                                        padding='max_length', max_length=128, truncation=True, return_tensors="pt").to(self.device)
                            temp_x = self.model(**tokenized_input)[0][:, 0, :]
                        x = torch.cat((x, temp_x.to(torch.device("cpu"))), dim=0) 
                        del tokenized_input
                        del temp_x

                        temp_y_a = torch.tensor([item_dict["label_a"]], dtype=torch.long)
                        y_a = torch.cat((y_a, temp_y_a), dim=0)
                
                        index_reply_to = temp_id_list.index(item_dict["in_reply_to_status_id"])
                        
                        temp_edge = torch.tensor([[index, index_reply_to]], dtype=torch.long)
                        
                        if edge_index is None:
                            edge_index = temp_edge
                        else:
                            edge_index = torch.cat((edge_index, temp_edge), dim=0)
                    except:
                        pass
        
            if(x is not None and edge_index is not None and y_a is not None and y_b is not None):
                torch.save(Data(x=x, edge_index=edge_index.t().contiguous(), y_a=y_a, y_b=y_b), osp.join(self.processed_dir, f'data_{self.idx}.pt'))
                self.idx += 1
            else:
                torch.save(Data(x=x, edge_index=torch.tensor([[0], [0]], dtype=torch.long), y_a=y_a, y_b=y_b), osp.join(self.processed_dir, f'data_{self.idx}.pt'))
                self.idx += 1
            
    def len(self):
        return len(self.belong_to_which_source_tweet_map)
        
    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data

class TestDataset(Dataset):

    def __init__(self, root, belong_to_which_source_tweet_map=None, config_path=None, model_path=None, checkpoint=None, device=None, transform=None, pre_transform=None, pre_filter=None):
        
        self.belong_to_which_source_tweet_map = belong_to_which_source_tweet_map
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.config = AutoConfig.from_pretrained(config_path)
        self.model = AutoModel.from_pretrained(model_path, config=self.config).to(device).eval()
        self.idx = 0
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return ["./processed_data/2017taskb_testing/raw/2017_testing_features_extraction.csv"]
    
    @property
    def processed_file_names(self):
        return []

    def _download(self):
        pass

    def process(self):
        for key in tqdm(self.belong_to_which_source_tweet_map):
            temp_id_list = []
            temp_text_list = []
            
            x = None
            edge_index = None
            y_a = None
            y_b = torch.tensor([ self.belong_to_which_source_tweet_map[key]["label"] ], dtype=torch.long)
            
            for index, item_dict in enumerate(self.belong_to_which_source_tweet_map[key]["item_dict_list"]):
                if(item_dict["twitter_id"] == item_dict["belong_to_which_source_tweet"]): 
                    temp_id_list.append(item_dict["twitter_id"])
                    temp_text_list.append(item_dict["text"])

                    with torch.no_grad():
                        tokenized_input = self.tokenizer(item_dict["text"], item_dict["text"], return_token_type_ids=True, 
                                                    padding='max_length', max_length=128, truncation=True, return_tensors="pt").to(device)
                        temp_x = self.model(**tokenized_input)[0][:, 0, :]
                    x = temp_x.to(torch.device("cpu")) 
                    del tokenized_input
                    del temp_x         
                    y_a = torch.tensor([item_dict["label_a"]], dtype=torch.long)

                else:
                    temp_id_list.append(item_dict["twitter_id"])
                    temp_text_list.append(item_dict["text"])

                    try: 
                        with torch.no_grad():
                            tokenized_input = self.tokenizer(temp_text_list[0], item_dict["text"], return_token_type_ids=True, 
                                                        padding='max_length', max_length=128, truncation=True, return_tensors="pt").to(device)
                            temp_x = self.model(**tokenized_input)[0][:, 0, :]
                        x = torch.cat((x, temp_x.to(torch.device("cpu"))), dim=0) 
                        del tokenized_input
                        del temp_x
                        temp_y_a = torch.tensor([item_dict["label_a"]], dtype=torch.long)
                        y_a = torch.cat((y_a, temp_y_a), dim=0)
                
                        index_reply_to = temp_id_list.index(item_dict["in_reply_to_status_id"])
                        
                        temp_edge = torch.tensor([[index, index_reply_to]], dtype=torch.long)
                        
                        if edge_index is None:
                            edge_index = temp_edge
                        else:
                            edge_index = torch.cat((edge_index, temp_edge), dim=0)
                    except:
                        pass
        
            if(x is not None and edge_index is not None and y_a is not None and y_b is not None):
                torch.save(Data(x=x, edge_index=edge_index.t().contiguous(), y_a=y_a, y_b=y_b), osp.join(self.processed_dir, f'data_{self.idx}.pt'))
                self.idx += 1
            else:
                torch.save(Data(x=x, edge_index=torch.tensor([[0], [0]], dtype=torch.long), y_a=y_a, y_b=y_b), osp.join(self.processed_dir, f'data_{self.idx}.pt'))
                self.idx += 1
    
    def len(self):
        return len(self.belong_to_which_source_tweet_map)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data
             