import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F

class Pure_Index(nn.Module):
    def __init__(self, cfg):
        super(Pure_Index, self).__init__()
        self.query_num = cfg.query_num
        self.feature_dim = cfg.feature_dim
        self.device = cfg.device
        self.cfg = cfg
        self.embedding = nn.Embedding(self.query_num, self.feature_dim).to(self.device)
        self.init_parameters()
        self.train_setting()
    
    def init_parameters(self):
        nn.init.xavier_normal_(self.embedding.weight)
    
    def train_setting(self):
        cfg = self.cfg.training_params
        l = [
            {'params': self.embedding.parameters()},
        ]
        self.optimizer = torch.optim.AdamW(l, lr=cfg.lr, betas=(cfg.min_betas, cfg.max_betas), weight_decay=cfg.weight_decay)
        warmup_steps = cfg.warmup_steps
        self.scheduler = lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=warmup_steps, T_mult=cfg.T_mult, eta_min=cfg.eta_min)
    
    def encode(self, image, step):
        logits = self.embedding.weight.unsqueeze(0)
        one_hot = F.gumbel_softmax(logits, tau=1, hard=True)
        index = torch.where(one_hot)
        one_grad = one_hot[index]

        one_start = torch.tensor([1]).to('cuda:0')
        one_end   = torch.tensor([1]).to('cuda:0')
        index_start = torch.tensor([self.feature_dim ]).to('cuda:0')
        index_end = torch.tensor([self.feature_dim + 1]).to('cuda:0')
                
        full_one_grad = torch.cat((one_start, one_grad, one_end), dim=0)
        full_index = torch.cat((index_start, index[2], index_end), dim=0)
        
        return full_one_grad, full_index