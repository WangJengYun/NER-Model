import os 
import logging
from datetime import datetime

class create_logger:
    def __init__(self,mode):
        filename = '{}_{:%Y-%m-%d_%H_%M_%S}.log'.format(mode,datetime.now()) 
        self.saving_file_path = self.get_file_path(filename)

    def get_file_path(self,filename):
        
        foldername = 'logs'
        if not os.path.exists(foldername):
            os.makedirs(foldername)
        return os.path.join(foldername,filename)
    
    def logger_init(self):
        logging.basicConfig(level= logging.DEBUG,#logging.INFO,
                            format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            handlers=[logging.FileHandler(self.saving_file_path, 'w', 'utf-8'),
                                      logging.StreamHandler()])

class ConfigLogger(object):
    def __init__(self, log):
        self.__log = log
    def __call__(self, config):
        self.__log.info("Config:")
        config.write(self)
    def write(self, data):
        # stripping the data makes the output nicer and avoids empty lines
        line = data.strip()
        self.__log.info(line)

class RunningAverage():
    """A simple class that maintains the running average of a quantity
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)

if __name__ == "__main__":
    pass 




