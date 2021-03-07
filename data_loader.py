import torch 
import configparser
from torch.utils.data import Dataset
from transformers import BertTokenizer

config = configparser.ConfigParser() 
config.read("model.conf")
tokenizer = BertTokenizer.from_pretrained(config['pytorch_model']['pytorch_Bert_pretrained_model'], do_lower_case=True)

tokenizer.tokenize('進入老年， 這 是 人 生 的 一 大 轉 折 ， 也 是 生 命 之 旅 的 最 後 階 段 。')