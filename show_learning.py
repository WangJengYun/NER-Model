import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from optimizer import get_training_optimizer
import configparser
import transformers
config = configparser.ConfigParser() 
config.optionxform = str
config.read("config.conf")

train_size = 5000
model = nn.Linear(10, 5)
optimizer,scheduler = get_training_optimizer(model.parameters(),
                                             train_size = train_size,
                                             config = config)

lrs = []
n_epochs = (train_size//int(config['hyperparameter']['batch_size']))* int(config['hyperparameter']['epoch_num'])
for i in range(n_epochs):
    optimizer.step()
    lrs.append(optimizer.param_groups[0]["lr"])
    scheduler.step()   
plt.plot(lrs)
plt.show()