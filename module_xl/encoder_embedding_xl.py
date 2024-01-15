import torch
import open_clip
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from module.component import Block, PositionalEncoding
import torch.optim.lr_scheduler as lr_scheduler

class Encoder_Embedding_XL(nn.Module):
    def __init__(self, cfg) -> None:
        super(Encoder_Embedding_XL, self).__init__()
        self.device = cfg.device
        self.cfg = cfg
        # prepare backbone
        # self.backbone, _ = open_clip.create_model_from_pretrained(
        #     model_name=cfg.clip_model_name,
        #     pretrained=cfg.clip_pretrained_model_name_or_path,
        # )
        self.backbone = CLIPModel.from_pretrained(cfg.clip_pretrained_model_name_or_path).vision_model
        self.backbone.to(cfg.device)
        
        # prepare attention pooler
        num_transformer_block = cfg.num_transformer_block
        block_list = []
        for i in range(num_transformer_block):
            block_list.append(Block(cfg.block).to(cfg.device))
        self.blocks = nn.ModuleList(block_list)
        
        # prepare query and positional embedding
        self.query_embed = nn.Embedding(cfg.query_num, cfg.query_dim).to(self.device)
        self.positional_embedding = PositionalEncoding(cfg.query_dim, cfg.query_num).to(self.device)
        
        # prepare final mlp layer
        self.to_embedding = nn.Linear(cfg.query_dim, cfg.feature_dim).to(self.device)
        
        self.freeze_backbone()
        self.init_parameters()
        self.train_setting()

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad_(False)  
    
    def init_parameters(self):
        nn.init.xavier_normal_(self.query_embed.weight)
        nn.init.xavier_normal_(self.to_embedding.weight)
        nn.init.zeros_(self.to_embedding.bias)

    def train_setting(self):
        l = [
            {'params': self.query_embed.parameters()},
            {'params': self.to_embedding.parameters()},
            {'params': self.blocks.parameters()},
        ]
        cfg = self.cfg.training_params
        self.optimizer = torch.optim.AdamW(l, lr=cfg.lr, betas=(cfg.min_betas, cfg.max_betas), weight_decay=cfg.weight_decay)
        warmup_steps = cfg.warmup_steps
        self.scheduler = lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=warmup_steps, T_mult=cfg.T_mult, eta_min=cfg.eta_min)
    
    def encode(self, image, step):
        B, _, _, _ = image.shape
        # feature = self.backbone(image.to(self.device))['image_embs']
        feature = self.backbone(pixel_values=image.to(self.device))['last_hidden_state'][:,1:,:]
        query = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        query = self.positional_embedding(query)
        for block in self.blocks:
            query = block(query, feature)
        query = self.to_embedding(query)
        return query