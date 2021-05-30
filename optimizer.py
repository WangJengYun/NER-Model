import re 
import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Adam

class Transformer_Adaw(Optimizer):
    def __init__(
        self,
        params,
        lr = 1e-3,
        betas = (0.9,0.999),
        eps = 1e-6,
        weight_decay = 0.0,
        correct_bias = True):

        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self,closure = None):

        loss = None 
        if closure is not None :
            loss  = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                
                if p.grad is None:
                    continue

                grad = p.grad.data 

                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                if len(state) == 0 :
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data) # Exponential moving average of gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)  # Exponential moving average of squared gradient values

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state['step'] += 1
                
                # in-place operations on the Tensors directly by appending an _ to the method name.
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1
                
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])
        
        return loss


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    num_cycles = 0.5,
    last_epoch = -1):

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

# get parameters
def get_optimizer_grouped_params(model,config):

    named_params = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    no_decay_reg = '.+({})'.format('|'.join(no_decay))

    optimizer_grouped_parameters = []
    if bool(config['hyperparameter']['full_fine_tuning']):
        
        # Bert 
        bert_name = [n for n,_ in named_params if 'bert' in n ]
        if bert_name:
            bert_nodecay_name = [n for n in bert_name if re.match(no_decay_reg,n)]
            bert_other_name = [n for n in bert_name if n not in bert_nodecay_name]
            optimizer_grouped_parameters.append({'params':[p for n,p in named_params if n in bert_other_name],
                                                 'weight_decay':float(config['hyperparameter']['weight_decay'])})
            optimizer_grouped_parameters.append({'params':[p for n,p in named_params if n in bert_nodecay_name],
                                                 'weight_decay':0.0 })
        
        # bilstm
        bilstm_name = [n for n,_ in named_params if 'bilstm' in n ]
        if bilstm_name:
            bilstm_nodecay_name = [n for n in bilstm_name if re.match(no_decay_reg,n)]
            bilstm_other_name = [n for n in bilstm_name if n not in bilstm_nodecay_name]
            optimizer_grouped_parameters.append({'params':[p for n,p in named_params if n in bilstm_other_name],
                                                 'weight_decay':float(config['hyperparameter']['weight_decay']),
                                                 'lr':float(config['hyperparameter']['learning_rate'])})
            optimizer_grouped_parameters.append({'params':[p for n,p in named_params if n in bilstm_nodecay_name],
                                                 'weight_decay':0.0,
                                                 'lr':float(config['hyperparameter']['learning_rate'])})
        # output                                                 
        output_name = [n for n,_ in named_params if 'linear_for_output' in n ]
        if output_name:
            output_nodecay_name = [n for n in output_name if re.match(no_decay_reg,n)]
            output_other_name = [n for n in output_name if n not in output_nodecay_name]
            optimizer_grouped_parameters.append({'params':[p for n,p in named_params if n in output_other_name],
                                                 'weight_decay':float(config['hyperparameter']['weight_decay']),
                                                 'lr':float(config['hyperparameter']['learning_rate'])})
            optimizer_grouped_parameters.append({'params':[p for n,p in named_params if n in output_nodecay_name],
                                                 'weight_decay':0.0,
                                                 'lr':float(config['hyperparameter']['learning_rate'])})
        # ctf                                            
        crf_name = [n for n,_ in named_params if 'crf' in n ]
        if crf_name:
            optimizer_grouped_parameters.append({'params':[p for n,p in named_params if n in crf_name],
                                                 'lr':float(config['hyperparameter']['learning_rate'])})
    else :
        pass
    
    return optimizer_grouped_parameters

# train_size = 16500   
def get_training_optimizer(optimizer_grouped_parameters,train_size,config):
    selected_optimizer = config['model']['selected_optimizer']
    selected_scheduler = config['model']['selected_scheduler']

    learning_rate = float(config['hyperparameter']['learning_rate'])
    epoch_num = int(config['hyperparameter']['epoch_num'])
    batch_size =  int(config['hyperparameter']['batch_size'])
    train_steps_per_epoch = train_size // batch_size

    optimizer = None
    scheduler = None
    if selected_optimizer == 'Transformer_Adaw':
        optimizer = Transformer_Adaw(params = optimizer_grouped_parameters,\
                     lr = learning_rate,
                     correct_bias = False)
    elif selected_optimizer == 'Adaw':
        optimizer = Adam(params = optimizer_grouped_parameters, lr =learning_rate)
    
    if selected_scheduler == 'cosine_schedule_with_warmup':
        scheduler  = get_cosine_schedule_with_warmup(optimizer,
                                              num_warmup_steps = epoch_num//10 * train_steps_per_epoch,
                                              num_training_steps = epoch_num * train_steps_per_epoch)
    elif selected_scheduler == 'lr_decay':
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(1 + 0.05*epoch))

    return optimizer,scheduler