import os 
import torch
import configparser
from transformers import BertModel,BertPreTrainedModel
from torch.utils.data import DataLoader
from torchsummary import summary
import torch.nn as nn
import  torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torchcrf import CRF
config = configparser.ConfigParser() 
config.optionxform = str
config.read("config.conf")

class NER_For_Bert(BertPreTrainedModel):
    def __init__(self,config):
        super(NER_For_Bert,self).__init__(config)

        self.num_labels = config.num_labels
        # BERT
        self.bert_block = BertModel(config)
        # LSTM
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.bilstm = nn.LSTM(input_size = config.lstm_embedding_size,
                          hidden_size = config.lstm_embedding_size,
                          batch_first = True,
                          num_layers = 2,
                          dropout = config.lstm_dropout_prob,
                          bidirectional = True)
        # ouput
        self.linear_for_output = nn.Linear(config.lstm_embedding_size*2,config.num_labels)
        # crf
        self.crf = CRF(config.num_labels,batch_first = True)

        self.init_weights()

    def forward(self,
                input_ids,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None):
        # Bert 
        bert_output = self.bert_block(input_ids = input_ids,
                                  attention_mask = attention_mask,
                                  token_type_ids = token_type_ids,
                                  position_ids = position_ids,
                                  head_mask = head_mask,
                                  inputs_embeds = inputs_embeds)[0]
    
        # bilstm
        bert_output = self.dropout(bert_output)
        lstm_output, _ = self.bilstm(bert_output)
    
        # output
        logits = self.linear_for_output(lstm_output)    

        # crf
        loss = None
        if labels is not None :
          loss = self.crf(logits,labels,attention_mask)
          loss = -loss

        return loss,logits


class BertForTokenClassification(BertPreTrainedModel):

    def __init__(self,config):
        super().__init__(config)

        self.num_labels = config.num_labels

        self.bert_block = BertModel(config,add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear_for_output = nn.Linear(config.hidden_size, config.num_labels)
  
    def forward(self,
                input_ids,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None):
    
        # Bert 
        bert_output = self.bert_block(input_ids = input_ids,
                                  attention_mask = attention_mask,
                                  token_type_ids = token_type_ids,
                                  position_ids = position_ids,
                                  head_mask = head_mask,
                                  inputs_embeds = inputs_embeds)[0]    
        # bert_output (batch,token_size,num_label)
        logits = self.linear_for_output(bert_output)    
        loss = None 
        if labels is not None :
          loss_function = nn.CrossEntropyLoss()
          active_loss = labels.gt(-1).view(-1)
          active_logits = logits.view(-1, self.num_labels)    
          active_labels = torch.where(active_loss,\
                                      labels.view(-1),\
                                      torch.tensor(loss_function.ignore_index).type_as(labels))    
          loss = loss_function(active_logits, active_labels)       
        return loss,logits           
    
def get_model(config):
    model = None 
    mode = config['model']['mode']
    device = config['env']['devive']
    selected_model = config['model']['selected_model']
    num_labels = int(config['model']['num_labels'])

    model_for_funtuning = config['pytorch_model']['pytorch_Bert_pretrained_model']
    model_for_perdict =  config['pytorch_model']['pytorch_Bert_model']
    
    # select model 
    if selected_model == 'NER_For_Bert':  
        model = NER_For_Bert
    elif selected_model == 'BertForTokenClassification':  
        model = BertForTokenClassification
    else: 
        raise ValueError('Not found this mode')
    
    # import model weight      
    if mode == 'training':
        model = model.from_pretrained(model_for_funtuning,num_labels = num_labels)
    
    elif mode == 'testing':
        model_file_path = os.path.join(model_for_perdict,'pytorch_model.pt')
        model = torch.load(model_file_path)
        model.eval()
    else :
      pass 

    # convert model to GPU
    model.to(device)

    return model
  

if __name__ == "__main__":
   pass
    # model = NER_For_Bert.from_pretrained(config['pytorch_model']['pytorch_Bert_pretrained_model'],\
    #                                        num_labels = len(config['Label_mapping']))

    # # model = BertForTokenClassification.from_pretrained(config['pytorch_model']['pytorch_Bert_pretrained_model'],\
    # #                                        num_labels = len(config['Label_mapping']))
    # model.to('cuda')
    # model(input_ids = input_ids,attention_mask =labels.gt(-1) ,labels = labels)
