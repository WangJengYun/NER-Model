import os 
import gc
import logging
import warnings
import configparser
import torch
import torch.nn as nn
from tqdm import tqdm,trange
from utils import RunningAverage
from datetime import datetime
from bert_model import NER_For_Bert
from optimizer import get_optimizer_grouped_params,get_training_optimizer
from metric import parsing_ner_format,Evaluator_for_ner
import numpy as np 
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class trainer:
    
    def __init__(self,model,config,logger):
        
        self.model = model
        self.config = config
        self.mode = None
        self.selected_model = config['model']['selected_model']
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.saving_model_folder = None
        
        self.train_data_size = None
        self.val_data_size = None
        self.test_data_size = None
        self.optimizer = None
        self.scheduler = None
        self.batch_size = int(config['hyperparameter']['batch_size'])
        self.label_padding_idx = int(config['Dataset']['label_padding_idx'])
        self.patience_num = int(config['hyperparameter']['patience_num'])
        self.patience_value = float(config['hyperparameter']['patience_value'])
        self.epoch_num = int(config['hyperparameter']['epoch_num'])
        self.clip_grad = int(config['hyperparameter']['clip_grad'])
        self.training_datatime = datetime.now()
        self.saving_model_path = config['training_setting']['saving_model_path']
        self.logger = logger


    def import_data(self,input_data,input_size):
        
        if len(input_data) == 2:
            self.mode = 'training'     
            self.train_data,self.val_data = input_data
            self.train_data_size,self.val_data_size = input_size
    
        elif len(input_data) == 1 :
            self.mode == 'testing'
            self.test_data = input_data
            self.test_data_size = input_size
        
        return self 
    
    def run(self):
        grouped_params = get_optimizer_grouped_params(self.model,self.config)
        self.optimizer,self.scheduler = get_training_optimizer(grouped_params,self.train_data_size,self.config)
        self.best_eval_metric = 0.0
        self.current_num_patience = 0 
        for epoch in range(1,self.epoch_num + 1):
            self.logger.info("===== Epoch {}=====".format(epoch))

            epoch_train_loss = self.train_epoch(epoch)
            
            val_metrics  = self.evaluate()
            self.logger.info("evaluate: {}".format(val_metrics))
            
            self.early_stopping(epoch,epoch_train_loss,val_metrics)
            self.logger.info("best_eval_metric = {};current_num_patience = {}".format(self.best_eval_metric, self.current_num_patience))

            if (self.current_num_patience == self.patience_num) or (epoch == self.epoch_num):
                self.logger.info("Best Epoch: {} ; Best train loss: {} ; Best val f1".format(epoch,epoch_train_loss,self.best_eval_metric))
                self.logger.info("Training Finished!")
                break

    def train_epoch(self,epoch):
        self.model.train()

        train_losses = 0 
        train_steps = self.train_data_size//self.batch_size
        loss_avg = RunningAverage()
        progress = trange(train_steps)
        training_data = list(self.train_data)
        for idx,_ in enumerate(progress):
        # for idx,batch_samples in enumerate(tqdm(self.train_data)):

            batch_idx, batch_labels = training_data[idx]
            # batch_idx, batch_labels = batch_samples

            loss,_ = self.model(input_ids = batch_idx,\
                                attention_mask = batch_labels.gt( self.label_padding_idx),\
                                labels = batch_labels)
            
            train_losses += loss.item()
            loss_avg.update(loss.item())

            # clear pevious gradients, compute gradients of all variable wrt loss 
            self.model.zero_grad()
            loss.backward()

            # gradient clipping
            nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.clip_grad)

            # performs updates using calculated gradients
            self.optimizer.step()
            self.scheduler.step()

            # # reducing memory
            # del loss,batch_idx, batch_labels
            # torch.cuda.empty_cache()

            progress.set_postfix(loss='{:05.6f}'.format(loss_avg()))
        
        # reducing memory
        del batch_idx, batch_labels,training_data
        torch.cuda.empty_cache()
            
        train_loss = float(train_losses)/train_steps 
        self.logger.info("Epoch: {}, train loss: {}".format(epoch, train_loss))

        return train_loss

    def early_stopping(self,epoch,epoch_train_loss,val_metrics):
        current_metric = val_metrics['full']
        
        if epoch == 1:
            self.best_eval_metric = current_metric
        else:
            if current_metric >= self.best_eval_metric:
                self.best_eval_metric = current_metric
                self.current_num_patience = 0
                self.saving_model(epoch,epoch_train_loss)
            else:
                self.current_num_patience += 1

    def evaluate(self):
        self.model.eval()
        eval_data = []
        eval_true_labels = []
        eval_pred_labels = []
        with torch.no_grad():
            for idx,batch_samples in enumerate(tqdm(self.val_data)):
                batch_idx, batch_labels = batch_samples
                label_masks = batch_labels.gt( self.label_padding_idx)
                loss,batch_output = self.model(input_ids = batch_idx,\
                                                attention_mask = label_masks,\
                                                labels = batch_labels)
                pred_batch_labels = self._decode(batch_output,label_masks)
                
                eval_data.extend(batch_idx.tolist())
                eval_true_labels.extend(batch_labels.tolist())
                eval_pred_labels.extend(pred_batch_labels)
        
        assert len(eval_true_labels) == len(eval_pred_labels)

        metrics = {}
        self.eval_true_labels_tuple = parsing_ner_format(eval_data,eval_true_labels,self.config).get_entities()
        self.eval_pred_labels_tuple = parsing_ner_format(eval_data,eval_pred_labels,self.config).get_entities()
        
        eval_result = Evaluator_for_ner(self.eval_true_labels_tuple,self.eval_pred_labels_tuple)

        metrics['full'] = eval_result.exact_metric('full')
        metrics['label_score'] = eval_result.exact_metric('label_score')
        return metrics
    
    def predict(self):
        pred_data = []
        pred_labels = []
        for idx,batch_samples in enumerate(tqdm(self.test_data)):
            batch_idx, batch_labels = batch_samples
            label_masks = batch_labels.gt(self.label_padding_idx)
            _,batch_output = self.model(input_ids = batch_idx,\
                                            attention_mask = label_masks)
            pred_batch_labels = self._decode(batch_output,label_masks)
            
            pred_data.extend(batch_idx.tolist())
            pred_labels.extend(pred_batch_labels)
        
        self.pred_labels_tuple = parsing_ner_format(pred_data,pred_labels,self.config).get_entities()
        return self.pred_labels_tuple

    def saving_model(self,epoch,epoch_train_loss):
        self.logger.info("***** Saving model *****")
        folder_name = 'checkpoint_{:%Y-%m-%d_%H_%M_%S}/'.format(self.training_datatime) 
        saving_path = os.path.join(self.saving_model_path,folder_name)

        if not os.path.exists(saving_path):
            os.makedirs(saving_path)
        saving_model_path = os.path.join(saving_path,'pytorch_model.pt')
        self.model.save_pretrained(saving_path)
        torch.save(self.model, saving_model_path)
    
    def _decode(self,pred_value,mask):
        if self.selected_model == 'NER_For_Bert':
            pred_batch_labels = self.model.crf.decode(pred_value, mask=mask)
        elif self.selected_model == 'BertForTokenClassification':
            batch_output_numpy = pred_value.detach().cpu().numpy()
            label_masks_numpy = mask.detach().cpu().numpy()
            pred_batch_labels = [np.argmax(target,axis = 1)[mask_Flag].tolist() for target, mask_Flag in list(zip(batch_output_numpy,label_masks_numpy))]
        else :
            pass 
        return pred_batch_labels


    
if __name__ == "__main__":
    pass
# 
# #-----------------------------
# #-----------------------------
# # from transformers import BertModel,BertPreTrainedModel
# # model = BertModel.from_pretrained(config['pytorch_model']['pytorch_Bert_pretrained_model'],num_labels=9)
# # model.to(config['env']['devive'])
# # bert_block = model
# 
# model = NER_For_Bert.from_pretrained(config['pytorch_model']['pytorch_Bert_pretrained_model'],num_labels=9)
# model.to(config['env']['devive'])
# 
# input_data, batch_labels = list(val_loader)[0]
# y_har = model(input_data,labels=batch_labels)


