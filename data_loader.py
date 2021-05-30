import os
import torch 
import logging
import random
import numpy as np 
import configparser
from torch.utils.data import Dataset
from transformers import BertTokenizer

class NER_Dataset(object):
    def __init__(self,config):

        self.tokenizer = BertTokenizer.from_pretrained(config['pytorch_model']['pytorch_Bert_pretrained_model'], do_lower_case=True)
        
        label_mapping = dict(config['Label_mapping'])
        self.label_to_idx = label_mapping
        self.idx_to_label = dict(zip(label_mapping.values(),label_mapping.keys()))
        
        self.dataset_path = config['Dataset']['Dataset_path']  
        self.is_cls_flag = bool(config['Dataset']['is_cls_flag'])   
        self.align_for_token_maxlen = bool(config['Dataset']['align_for_token_maxlen'])   
        self.token_padding_idx = int(config['Dataset']['token_padding_idx'])
        self.cls_padding_idx = int(config['Dataset']['cls_padding_idx'])
        self.label_padding_idx = int(config['Dataset']['label_padding_idx'])
        self.token_max_len = int(config['Dataset']['token_max_len'])
        self.sample_seed = int(config['Dataset']['sample_seed'])

        self.device = config['env']['devive']

        self.data = None 
        self.data_size = None
        self.datatype = None

    def load_data(self,datatype,input_sentences = None):
        # datatype = 'val'
        self.datatype = datatype
        sentences = None
        label= None
        if input_sentences == None:
            sentences_path = os.path.join(self.dataset_path,self.datatype,'sentences.txt')
            sentences = open(sentences_path,encoding='utf-8').readlines()
            sentences = [s.strip() for s in sentences]
            
            if self.datatype in ['train','val']:
                
                label_path = os.path.join(self.dataset_path,self.datatype,'target.txt')
                label = open(label_path,encoding='utf-8').readlines()    
                label = [s.strip() for s in label]

                assert len(sentences) == len(label)
 
        else:
            sentences = input_sentences

        self.data_size = len(sentences)

        self.data = self.convert_word_to_idx(sentences,label)

        return self
    
    def convert_word_to_idx(self,sentences,label):
        add_word = 0
        if self.is_cls_flag:
            add_word = 1
            sentences = [ '[CLS]' + s for s in sentences]

        
        token_idx = []
        label_idx = []
        for idx in range(self.data_size):
            # handling sentence
            singel_words_list = self.tokenizer.tokenize(sentences[idx])
            token_idx.append(self.tokenizer.convert_tokens_to_ids(singel_words_list))
            
            # handling label
            if self.datatype in ['train','val']:
                current_label = label[idx].strip().split(' ')
            else :
                current_label = ['O']*(len(singel_words_list)-add_word)
            label_idx.append([self.label_to_idx[i] for i in current_label])
        
        for i in range(self.data_size):
            assert len(token_idx[i]) == (len(label_idx[i]) + add_word)

        return list(zip(token_idx,label_idx))
    
    def __getitem__(self,idx):
        return self.data[idx]
    
    def __len__(self):
        return self.data_size
    # batch = dataset[0:8]
    def collate_fn(self,batch):
        token_idx = [row[0] for row in batch]
        label_idx = [row[1] for row in batch]

        batch_len = len(token_idx)
        batch_token_maxlen = max([len(s) for s in token_idx])
        
        maxlen = self.token_max_len
        if (self.align_for_token_maxlen) and (batch_token_maxlen < maxlen):
            maxlen = batch_token_maxlen 
        
        batch_idx = self.token_padding_idx * np.ones((batch_len, maxlen))
        batch_label = self.label_padding_idx * np.ones((batch_len, maxlen))
        batch_pos = np.zeros((batch_len, maxlen))
        
        label_start_idx = 0
        if self.is_cls_flag:
            label_start_idx = 1
            batch_label[:,0] = self.cls_padding_idx

        for idx in range(batch_len):
            current_len = len(token_idx[idx])
            if current_len<= maxlen:
                batch_idx[idx][:current_len] = token_idx[idx]
                batch_label[idx][label_start_idx:current_len] = label_idx[idx]
            else :
                batch_idx[idx] = token_idx[idx][:maxlen]
                batch_label[idx][label_start_idx:] = label_idx[idx][:(maxlen-label_start_idx)]

        # since all data are indices, we convert them to torch LongTensors
        batch_idx = torch.tensor(batch_idx, dtype=torch.long)
        batch_label = torch.tensor(batch_label, dtype=torch.long)  
        batch_pos = torch.tensor(batch_pos, dtype=torch.long)  
        
        # shift tensors to GPU if available
        batch_idx, batch_label = batch_idx.to(self.device), batch_label.to(self.device)

        return batch_idx, batch_label
    # batch_size = 16
    def data_iterator(self,batch_size,shuffle = False):
        
        token_idx,label_idx = self.data

        order = list(range(self.data_size))
        if shuffle:
            random.seed(self.sample_seed)
            random.shuffle(order)
        
        for idx in range(self.data_size//batch_size):
            start_idx = idx*batch_size
            end_idx = (idx+1)*batch_size

            batch_data = (token_idx[start_idx:end_idx],label_idx[start_idx:end_idx])

            yield self.collate_fn(batch_data)

if __name__ == "__main__":
    pass 
    # import configparser
    # config = configparser.ConfigParser() 
    # config.optionxform = str
    # config.read("config.conf")
    # 
    # dataset = NER_Dataset(config).load_data('val')
# 
    # AA = dataset.data_iterator(batch_size = 8,shuffle = False)
# 
    # input_ids,labels= next(AA)