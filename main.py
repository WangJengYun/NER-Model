import logging
import warnings
import configparser
from utils import create_logger,ConfigLogger
from data_loader import NER_Dataset
from torch.utils.data import DataLoader
from train import trainer 
from bert_model import get_model
warnings.filterwarnings('ignore')

config = configparser.ConfigParser() 
config.optionxform = str
config.read("config.conf")

if __name__ == "__main__":
    create_logger(config['model']['mode']).logger_init()
    logger = logging.getLogger(__name__)

    logger.info('===== import configuration =====')
    config_logger = ConfigLogger(logging)
    config_logger(config)
    
    logger.info('===== import data =====')
    batch_size = int(config['hyperparameter']['batch_size'])

    train_dataset = NER_Dataset(config).load_data('train')
    train_loader = DataLoader(train_dataset, batch_size = batch_size,
                             shuffle=True,#num_workers = 8,pin_memory = True,
                             collate_fn=train_dataset.collate_fn)

    val_dataset = NER_Dataset(config).load_data('val')
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                             shuffle=True,#num_workers = 8,pin_memory = True,
                             collate_fn=val_dataset.collate_fn)

    logger.info('===== getting model =====')
    model = get_model(config = config)


    logger.info('===== Starting Training =====')
    result = trainer(model,config,logger)
    result.import_data((train_loader,val_loader),(len(train_dataset),len(val_dataset)))
    result.run()
    # result.evaluate()

    # input_data,labels = list(train_loader)[0]

