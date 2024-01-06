import torch.nn as nn
from module.pure_embedding import Pure_Embedding
from module.pure_index import Pure_Index
from module.decoder_embedding import Decoder_Embedding
from module.encoder_embedding import Encoder_Embedding
from module.decoder_index_small import Decoder_Index_Small

class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.cfg = cfg
        if cfg.encoder_type == 'pure_embedding':
            self.encoder = Pure_Embedding(cfg.pure_embedding)
        elif cfg.encoder_type == 'encoder_embedding':
            self.encoder = Encoder_Embedding(cfg.encoder_embedding)
        elif cfg.encoder_type == 'pure_index':
            self.encoder = Pure_Index(cfg.pure_index)
            
        if cfg.decoder_type == 'decoder_embedding':
            self.decoder = Decoder_Embedding(cfg.decoder_embedding)
        elif cfg.decoder_type == 'decode_index_small':
            self.decoder = Decoder_Index_Small(cfg.decode_index_small)
        
        if cfg.encoder_type == 'pure_embedding' or cfg.encoder_type == 'encoder_embedding':
            self.mode = 'embedding'
        else:
            self.mode = 'index'
    
    def encoder_decoder_embedding(self, batch, step):
        prompt_embeds = self.encoder.encode(batch['encode_pixel'], step)
        loss = self.decoder.decode(batch['decode_pixel'], prompt_embeds)
        return loss
    
    def inference_embedding(self, batch, step):
        prompt_embeds = self.encoder.encode(batch['encode_pixel'], step)
        image = self.decoder.inference_embedding(prompt_embeds)
        return image
    
    def encoder_decoder_index(self, batch, step):
        one_grad, txt_indices = self.encoder.encode(batch['encode_pixel'], step)
        loss = self.decoder.decode(batch['decode_pixel'], one_grad, txt_indices)
        return loss
    
    def inference_index(self, batch, step):
        one_grad, txt_indices = self.encoder.encode(batch['encode_pixel'], step)
        image = self.decoder.inference_index(one_grad, txt_indices)
        return image
    
    def encoder_decoder(self, batch, step):
        if self.mode == 'embedding':
            loss = self.encoder_decoder_embedding(batch, step)
        elif self.mode == 'index':
            loss = self.encoder_decoder_index(batch, step)
        loss.backward()
        self.encoder.optimizer.step()
        self.encoder.scheduler.step()
        self.encoder.optimizer.zero_grad()
        return loss.item()
    
    def inference(self, batch, step):
        if self.mode == 'embedding':
            image = self.inference_embedding(batch, step)
        elif self.mode == 'index':
            image = self.inference_index(batch, step)
        return image