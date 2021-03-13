import torch 
import numpy as np 
import configparser
from torch.utils.data import Dataset
from transformers import BertTokenizer

class NER_dataset(Dataset):
    def __init__(self,config,datatype,input_data = None):
        
        self.tokenizer = BertTokenizer.from_pretrained(config['pytorch_model']['pytorch_Bert_pretrained_model'], do_lower_case=True)
        self.label_to_id = dict(config['Label_mapping'])
        self.dataset_path = config['Dataset']['Dataset_path']
        
        self.dataset = self.convert_dataset_to_bertinput(datatype,input_data)

    def convert_dataset_to_bertinput(self,datatype,input_data):

        if input_data :
            sentences,target = input_data

        elif datatype :
            sentences = open(self.dataset_path + '/' + datatype + "/sentences.txt", "r",encoding='utf-8').readlines()
            targets = open(self.dataset_path + '/' + datatype + "/target.txt", "r",encoding='utf-8').readlines()
            assert len(sentences) == len(targets)
        else : 
            raise  ValueError('Please input "datatype" or "input_data"')
        
        input_data = []
        label = []
        for idx in range(len(sentences)) :
            # idx = 69
            token = ['[CLS]'] + self.tokenizer.tokenize(sentences[idx].strip())
            target = targets[idx].strip().split(' ')
            assert len(token) == len(target) + 1
            
            token_idx = self.tokenizer.convert_tokens_to_ids(token)
            token_pos = list(range(1,len(token) +1))
            target_idx = [self.label_to_id[c] for c in target]

            label.append(target_idx)
            input_data.append((token,token_idx,token_pos))
        
        return list(zip(input_data,label))

    def __getitem__(self,idx):
        # input_data = [row[0] for row in self.dataset[idx]]
        # label = [row[1] for row in self.dataset[idx]]
        batch = self.dataset[idx]
        return batch

    def __len__(self):
        return len(self.dataset)

    def collate_fn(self,batch):
        token_padding_idx = int(self.config['Dataset']['token_padding_idx'])
        label_padding_idx = int(self.config['Dataset']['label_padding_idx'])
        token_max_len = int(self.config['Dataset']['token_max_len'])

        input_data = [row[0] for row in batch]
        labels = [row[1] for row in batch]
        assert len(input_data) == len(labels)

        batch_len = len(input_data)

        token_max_len = max([len(data[0]) for data in input_data])

        batch_data = np.full((batch_len,token_max_len),token_padding_idx)
        batch_label_Flag = np.full((batch_len,token_max_len),0)
        batch_labels = np.full((batch_len,token_max_len),label_padding_idx)

        for idx in range(batch_len):
            _,token_idx,token_pos = input_data[idx]
            label = labels[idx]
            current_len = len(token_idx)
            if current_len <= token_max_len:
                batch_data[idx][:current_len] = token_idx
                input_token_pos = [ i-1 for i in token_pos if i-1 != 0 ]
                batch_label_Flag[idx][input_token_pos] = 1  
                batch_labels[idx][1:current_len] = label
           
        batch_data = torch.tensor(batch_data, dtype=torch.long)
        batch_label_Flag = torch.tensor(batch_label_Flag, dtype=torch.long)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)

        batch_data = batch_data.to(self.config['env']['devive'])
        batch_label_Flag = batch_label_Flag.to(self.config['env']['devive'])
        batch_labels = batch_labels.to(self.config['env']['devive'])

        return [batch_data, batch_label_Flag, batch_labels]

if __name__ == "__main__":
    pass        