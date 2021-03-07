import torch
import shutil
from transformers import BertConfig, BertForPreTraining, load_tf_weights_in_bert
from transformers.utils import logging

import configparser
logging.set_verbosity_info()
 

def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path):
    # Initialise PyTorch model
    config = BertConfig.from_json_file(bert_config_file)
    print("Building PyTorch model from configuration: {}".format(str(config)))
    model = BertForPreTraining(config)

    # Load weights from tf checkpoint
    load_tf_weights_in_bert(model, config, tf_checkpoint_path)

    # Save pytorch-model
    print("Save PyTorch model to {}".format(pytorch_dump_path))
    torch.save(model.state_dict(), pytorch_dump_path)


if __name__ == "__main__":
    config = configparser.ConfigParser() 
    config.read("model.conf")

    convert_tf_checkpoint_to_pytorch(config['TF_model']['TF_Bert_checkpoint'],\
                                     config['TF_model']['TF_Bert_config'],\
                                     config['pytorch_model']['pytorch_Bert_pretrained_model'])
    # copy vocab.txt
    shutil.copyfile(config['TF_model']['TF_Bert_vocab'],\
                    config['pytorch_model']['pytorch_Bert_vocab'])