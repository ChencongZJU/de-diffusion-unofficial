import os
import torch
import torch.nn as nn
from module_xl.pure_embedding_xl import Pure_Embedding_XL
from module_xl.encoder_embedding_xl import Encoder_Embedding_XL
from module_xl.decoder_embedding_xl import Decoder_Embedding_XL
from module_xl.pure_index_xl import Pure_Index_XL
from module_xl.decoder_index_xl import Decoder_Index_XL
from module_xl.decoder_index_small_xl import Decoder_Index_Small_XL
from module_xl.encoder_index_xl import Encoder_Index_XL

class Model_XL(nn.Module):
    def __init__(self, cfg) -> None:
        super(Model_XL, self).__init__()
        self.cfg = cfg
        if cfg.encoder_type == 'pure_embedding_xl':
            self.encoder = Pure_Embedding_XL(cfg.pure_embedding_xl)
        elif cfg.encoder_type == 'encoder_embedding_xl':
            self.encoder = Encoder_Embedding_XL(cfg.encoder_embedding_xl)
        elif cfg.encoder_type == 'pure_index_xl':
            self.encoder = Pure_Index_XL(cfg.pure_index_xl)
        elif cfg.encoder_type == 'encoder_index_xl':
            self.encoder = Encoder_Index_XL(cfg.encoder_index_xl)
        
        if cfg.decoder_type == 'decoder_embedding_xl':
            self.decoder = Decoder_Embedding_XL(cfg.decoder_embedding_xl)
        elif cfg.decoder_type == 'decoder_index_xl':
            self.decoder = Decoder_Index_XL(cfg.decoder_index_xl)
        elif cfg.decoder_type == 'decoder_index_small_xl':
            self.decoder = Decoder_Index_Small_XL(cfg.decoder_index_small_xl)
        
        if cfg.encoder_type == 'pure_embedding_xl' or cfg.encoder_type == 'encoder_embedding_xl':
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
        with torch.no_grad():
            if self.mode == 'embedding':
                image = self.inference_embedding(batch, step)
            elif self.mode == 'index':
                image = self.inference_index(batch, step)
            return image

    def save_embedding_embedding(self, batch, step):
        return self.encoder.encode(batch['encode_pixel'], step)

    def save_embedding(self, dir, batch, step):
        if self.mode == 'embedding':
            embedding = self.save_embedding_embedding(batch, step)
            torch.save(embedding, os.path.join(dir, f'{step}.pt'))