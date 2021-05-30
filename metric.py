import re 
import torch
import configparser
import numpy as np 
from collections import namedtuple
from transformers import BertTokenizer
from transformers import BertModel,BertPreTrainedModel
from torch.utils.data import DataLoader
from torchsummary import summary
import torch.nn as nn 
from torch.nn.utils.rnn import pad_sequence
from torchcrf import CRF
config = configparser.ConfigParser() 
config.optionxform = str
config.read("config.conf")

    
class parsing_ner_format:

    def __init__(self,batch_seqs,batch_labels,config,is_decoding = True):
        
        self.label_to_id =dict(map(reversed, dict(config['Label_mapping']).items())) 
        self.tokenizer = BertTokenizer.from_pretrained(config['pytorch_model']['pytorch_Bert_pretrained_model'], do_lower_case=True)
        
        if is_decoding:
            self.batch_seqs,self.batch_labels = self._decoding(batch_seqs,batch_labels)
        else: 
            self.batch_seqs,self.batch_labels = batch_seqs,batch_labels
        self.is_cls_flag = bool(config['Dataset']['is_cls_flag']) 

    def _decoding(self,batch_seqs,batch_labels):

        if type(batch_seqs).__name__ =='Tensor':
            batch_seqs_list = batch_seqs.tolist()
        else :
            batch_seqs_list = batch_seqs
        
        decode_batch_seqs_list = []
        for seq in batch_seqs_list :
            decode_batch_seqs_list.append([self.tokenizer.convert_ids_to_tokens(i) for i in seq if i not in [0,101]])

        
        if  type(batch_labels).__name__ =='Tensor':
            batch_labels_list = batch_labels.data.tolist()
        else : 
            batch_labels_list = batch_labels  

        decode_batch_labels_list = []
        for seq in batch_labels_list :
            decode_batch_labels_list.append([self.label_to_id[str(i)] for i in seq if i != -1])
  
        return decode_batch_seqs_list,decode_batch_labels_list
    
    def get_entities(self):
        assert len(self.batch_seqs) == len(self.batch_labels)
        batch_chunks = []
        for i in range(len(self.batch_labels)):
            sentence,labels_seq = self.batch_seqs[i],self.batch_labels[i]
            if self.is_cls_flag :
                labels_seq = labels_seq[1:]
            try:
                assert len(sentence) == len(labels_seq)
            except:
                print(sentence)
                print(labels_seq)
                print(i) 
                raise ValueError("SSS")

            seq_len = len(labels_seq)

            start_idx = None 
            end_idx = None
            chunks_type = None
            chunks = []
            for idx in range(seq_len):
                if self.check_end_of_chunk(idx,labels_seq):
                    end_idx = idx 
                    chunks.append((''.join(sentence[start_idx:end_idx]),
                                   chunks_type,(start_idx,end_idx)))
                if self.check_start_of_chunk(idx,labels_seq):        
                    start_idx = idx
                    chunks_type = labels_seq[idx].split('-')[-1]
            
            batch_chunks.append(chunks)
        return batch_chunks

    def check_start_of_chunk(self,idx,labels_seq):
        current_label = labels_seq[idx]
        if idx == 0:
            if bool(re.match('^B|^I',current_label)):
                return True
            else :
                return False
        else:
            previous_label = labels_seq[(idx-1)]
            cond1 = bool(re.match('^B',current_label))
            cond2 = (previous_label == 'O') and bool(re.match('^I',current_label))
            cond3 = (current_label!= 'O') and (previous_label.split('-')[-1] != current_label.split('-')[-1])

            if cond1|cond2|cond3:
                return True
            else :
                return False

    def check_end_of_chunk(self,idx,labels_seq): 
        current_label = labels_seq[idx]
        if idx == 0:
            return False
        else:
            previous_label = labels_seq[(idx-1)]
            cond1 = bool(re.match('^B',previous_label)) and current_label == 'O'
            cond2 = bool(re.match('^I',previous_label)) and  current_label == 'O'
            cond3 = bool(re.match('^B',previous_label)) and bool(re.match('^B',current_label))
            cond4 = (previous_label!= 'O') and (current_label!= 'O') \
                    and (previous_label.split('-')[-1] != current_label.split('-')[-1])

            if cond1|cond2|cond3|cond4:
                return True
            else :
                return False


class Evaluator_for_ner:
   
    def __init__(self,true_values,pred_values):
        
        if len(true_values) != len(pred_values):
            raise ValueError('Number of predicted documents does not equal true')

        self.Entity = namedtuple("Entity", "word type start_offset end_offset")
        self.true_values = self.convert_to_namedtuple(true_values)
        self.pred_values = self.convert_to_namedtuple(pred_values)
        
        self.label_types = config['Label']['label_type'].split(',')

    def convert_to_namedtuple(self,input_data):
        ouput_data = []
        for row in input_data:
            named_entities = []
            for word,e_type,(start_offset,end_offset) in row:
                named_entities.append(self.Entity(word, e_type, start_offset,end_offset))
            ouput_data.append(tuple(named_entities))
        return ouput_data
    

    def exact_metric(self,mode):

        def convert_array(entity):
            entity_array = None 
            if set(entity) == {()}:
                entity_array = np.zeros(len(entity))
            else: 
                entity_array = np.array(entity)
            return entity_array

        def _get_entity(data,label_type):
            ouput = []
            for row in data:
                single_ouput = [ entity for entity in row if entity.type == label_type]
                ouput.append(tuple(single_ouput))
            return ouput

        def get_f1_score(true_entity,pred_entity):
            n_correct = np.sum(convert_array(true_entity) == convert_array(pred_entity))
            n_true_entity = len(true_entity)
            n_pred_entity = len(pred_entity)

            p_value = n_correct / n_true_entity if n_true_entity > 0 else 0
            r_value = n_correct / n_pred_entity if n_pred_entity > 0 else 0

            score = 2 * p_value * r_value / (p_value + r_value) if p_value + r_value > 0 else 0
            return score

        ouput_score = None         
        if mode == 'full':
            ouput_score = get_f1_score(self.true_values,self.pred_values)
        elif mode == 'label_score':
            ouput_score = {}
            for label_type in self.label_types:

                true_label_entity = _get_entity(self.true_values,label_type)
                pred_label_entity = _get_entity(self.pred_values,label_type)
                ouput_score[label_type] = get_f1_score(true_label_entity,pred_label_entity)
        
        return ouput_score


if __name__ == "__main__":
    input_data,labels = list(train_loader)[0]
    batch_data = input_data[0]
    batch_labels = labels
    batch_idx_list,batch_labels_list = batch_data.tolist(),batch_labels.data.tolist() 

    result = parsing_ner_format(batch_idx_list,batch_labels_list).get_entities()
    
    AA,BB = parsing_ner_format(batch_idx_list,batch_labels_list)._decoding(batch_idx_list,batch_labels_list)

    Evaluator_for_ner(result,result).exact_metric('label_score')
