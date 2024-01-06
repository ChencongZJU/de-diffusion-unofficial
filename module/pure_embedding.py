import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

class Pure_Embedding(nn.Module):
    def __init__(self, cfg):
        super(Pure_Embedding, self).__init__()
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
        return self.embedding.weight.unsqueeze(0)