import os
import hydra
from omegaconf import OmegaConf
from rich.console import Console
from module.model import Model
from torch.utils.data import DataLoader
import open_clip
from tqdm import tqdm
import torch
import datetime
import wandb
from module.model import Model
from pathlib import Path
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from PIL import Image
from torchvision import transforms
from utils.utils import seed_everything
from utils.misc import(
    step_check,
    C,
    get_file_list,
    dict_to_device,
    to_primitive,
    dump_config,
    huggingface_offline,
    get_current_cmd
)
import random
import numpy as np

class Trainer():
    def __init__(self, cfg) -> None:
        self.device = cfg.device
        self.cfg = cfg
    
        # prepare model
        self.model = Model(cfg.model)
        
        # prepare dataset:
        self.prepare_transforms()
        if cfg.mode.train_single_image:
            self.prepare_single_image()
        
        # prepare save dir and wandb
        self.init_dir_wandb(cfg)

    def init_dir_wandb(self, cfg):
        day_timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
        hms_timestamp = datetime.datetime.now().strftime("%H%M%S")
        timestamp = f"{hms_timestamp}|{day_timestamp}"
        self.timestamp = timestamp
        uid = f"{timestamp}"
        
        self.save_dir = Path(f"./checkpoints/{day_timestamp}/{hms_timestamp}")
        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True, exist_ok=True)

        if cfg.use_wandb:
            wandb.init(
                project="dediffusion",
                entity='chencong2021',
                name=uid,
                config=to_primitive(cfg),
                sync_tensorboard=True,
                # magic=True,
                save_code=True,
            )
            
    def prepare_transforms(self):
        def _convert_to_rgb(image):
            return image.convert('RGB')
        self.encode_transform = transforms.Compose(
            [
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=None),
                transforms.CenterCrop(size=(224, 224)),
                _convert_to_rgb,
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
            ]
        )
        self.decode_transform = transforms.Compose(
            [
                transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(512),
                _convert_to_rgb,
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def prepare_single_image(self):
        image_dir = self.cfg.single_image.path
        image = Image.open(image_dir)
        # image = image.convert("RGB")
        encode_image = self.encode_transform(image).to(self.device).unsqueeze(0)
        decode_image = self.decode_transform(image).to(self.device).unsqueeze(0)
        self.single_batch = {
            'encode_pixel': encode_image,
            'decode_pixel': decode_image
        }
    
    def train_single_image(self):
        self.step = 0
        self.max_steps = self.cfg.train_setting.max_steps
        with tqdm(total=self.max_steps) as pbar:
            while self.step < self.max_steps:
                loss = self.model.encoder_decoder(self.single_batch, self.step)
                self.step = self.step + 1
                if self.step >= self.max_steps:
                    break
                if self.step % self.cfg.train_setting.inference_steps == 0:
                    image = self.model.inference(self.single_batch, self.step)[0]
                    image.save(os.path.join(self.save_dir, f'{self.step}.png'))
                if self.cfg.use_wandb:
                    wandb.log({
                        "loss": loss
                    }, step=self.step)
                
                pbar.set_description(f"Iter: {self.step}/{self.max_steps}")
                pbar.set_postfix(loss=f"{loss:.4f}")
                pbar.update(1)
                

@hydra.main(version_base="1.3", config_path="conf", config_name="train")
def main(cfg):
    seed_everything(0)
    trainer = Trainer(cfg)
    if cfg.mode.is_train:
        if cfg.mode.train_single_image:
            trainer.train_single_image()

if __name__ == "__main__":
    main()
