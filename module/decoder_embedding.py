import copy
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler, StableDiffusionPipeline, PNDMScheduler
from diffusers.image_processor import VaeImageProcessor


class Decoder_Embedding():
    def __init__(self, cfg) -> None:
        super(Decoder_Embedding, self).__init__()
        self.device = cfg.device
        self.cfg = cfg
        
        self.weights_dtype = (
            torch.float16 if cfg.half_precision_weights else torch.float32
        )
        pipe = StableDiffusionPipeline.from_pretrained(
            cfg.unet_pretrained_model_name_or_path,
            torch_dtype = self.weights_dtype
        ).to(self.device)
        
        with torch.no_grad():
            self.negative_prompt_embeds = pipe.encode_prompt("", 'cuda:0', 1, False)[0]
        
        ## init components
        self.text_encoder = pipe.text_encoder
        self.tokenizer = pipe.tokenizer
        self.unet = copy.deepcopy(pipe.unet)
        self.scheduler = DDPMScheduler.from_pretrained(cfg.unet_pretrained_model_name_or_path, subfolder="scheduler")
        self.vae = pipe.vae
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.do_classifier_free_guidance = cfg.do_classifier_free_guidance
        self.freeze()
    
    def freeze(self):
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)
        for p in self.vae.parameters():
            p.requires_grad_(False)
    
    def inference_embedding(self, prompt_embeds, num_inference_steps=50, guidance_scale = 7.5):
        # Prepare latents
        latents = torch.randn(1, 4, 64, 64).to('cuda:0')
        
        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        
        # Prepare embedding
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([self.negative_prompt_embeds, prompt_embeds])

        with torch.no_grad():
            for i, t in enumerate(timesteps):
                
                latent_model_input = torch.cat([latents] * 2)  if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                )[0]
                
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)                
                
                latents = self.scheduler.step(noise_pred, t, latents)[0]

        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        do_denormalize = [True] * image.shape[0]
        image = self.image_processor.postprocess(image, output_type='pil', do_denormalize=do_denormalize)
        return image
    
    def decode(self, images, embedding):
        # Convert images to latent space
        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        
        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()
        
        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
    
        if self.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.scheduler.config.prediction_type}")
        
        # Predict the noise residual and compute loss
        model_pred = self.unet(noisy_latents, timesteps, embedding).sample
        
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        loss = loss.mean()
        return loss