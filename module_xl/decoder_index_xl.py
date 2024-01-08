import copy
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler, StableDiffusionXLPipeline, StableDiffusionPipeline, PNDMScheduler
from diffusers.image_processor import VaeImageProcessor
from utils.utils import seed_everything

class Decoder_Index_XL():
    def __init__(self, cfg) -> None:
        super(Decoder_Index_XL, self).__init__()
        self.device = cfg.device
        self.cfg = cfg
        self.original_size = (cfg.original_size, cfg.original_size)
        self.target_size = (cfg.target_size, cfg.target_size)
        self.crops_coords_top_left = (0, 0)
        
        self.weights_dtype = (
            torch.float16 if cfg.half_precision_weights else torch.float32
        )
        pipe = StableDiffusionXLPipeline.from_pretrained(
            cfg.unet_pretrained_model_name_or_path,
            torch_dtype = self.weights_dtype
        ).to(self.device)
        
        ## prepare component
        self.text_encoder = pipe.text_encoder
        self.tokenizer = pipe.tokenizer
        self.text_encoder_2 = pipe.text_encoder_2
        self.tokenizer_2 = pipe.tokenizer_2
        self.scheduler = DDPMScheduler.from_pretrained(cfg.unet_pretrained_model_name_or_path, subfolder="scheduler")
        self.vae = pipe.vae
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.unet = pipe.unet
        del pipe
        self.do_classifier_free_guidance = cfg.do_classifier_free_guidance
        self.freeze()
        self.prepare_nagative()
    
    def freeze(self):
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)
        for p in self.text_encoder_2.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)
        for p in self.vae.parameters():
            p.requires_grad_(False)
    
    def prepare_nagative(self):
        negative_prompt_embeds_list = []
        negative_prompt = ''
        negative_prompts = [negative_prompt, negative_prompt]
        tokenizers = [self.tokenizer, self.tokenizer_2]
        text_encoders = [self.text_encoder, self.text_encoder_2]
        with torch.no_grad():
            for negative_prompt, tokenizer, text_encoder in zip(negative_prompts, tokenizers, text_encoders):
                text_inputs = tokenizer(
                    negative_prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                text_input_ids = text_inputs.input_ids
                untruncated_ids = tokenizer(negative_prompt, padding="longest", return_tensors="pt").input_ids

                negative_prompt_embeds = text_encoder(text_input_ids.to(self.device), output_hidden_states=True)

                # We are only ALWAYS interested in the pooled output of the final text encoder
                negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]

                negative_prompt_embeds_list.append(negative_prompt_embeds)
            
            self.negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

    def inference_index(self, one_grad, token_indices, num_inference_steps=30, guidance_scale = 7.5):
        # Prepare latents
        latents = torch.randn(1, 4, 128, 128).to('cuda:0')
        
        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        
        # prepare embedding
        prompt_embeds_list = []
        
        # text encoder1
        prompt_embeds = self.text_encoder(token_indices.unsqueeze(0).to(self.device), output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        prompt_embeds = torch.mul(one_grad.unsqueeze(0).unsqueeze(-1), prompt_embeds)
        prompt_embeds_list.append(prompt_embeds)
        
        # text encoder2
        prompt_embeds = self.text_encoder_2(token_indices.unsqueeze(0).to(self.device), output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        prompt_embeds = torch.mul(one_grad.unsqueeze(0).unsqueeze(-1), prompt_embeds)
        prompt_embeds_list.append(prompt_embeds)
        
        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        
        add_text_embeds = pooled_prompt_embeds
        negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        
        add_time_ids = torch.tensor(list(self.original_size + self.crops_coords_top_left + self.target_size)).to(self.device)
        negative_add_time_ids = add_time_ids
        
        # Prepare CFG
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([self.negative_prompt_embeds, prompt_embeds])
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)
        
        with torch.no_grad():
            for i, t in enumerate(timesteps):
                
                latent_model_input = torch.cat([latents] * 2)  if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                # predict the noise residual
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=None,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]
                
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                # noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                latents = self.scheduler.step(noise_pred, t, latents)[0]

        latents = latents.to(dtype=torch.float32)
        self.vae.to(dtype=torch.float32)
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        do_denormalize = [True] * image.shape[0]
        image = self.image_processor.postprocess(image.detach(), output_type='pil', do_denormalize=do_denormalize)
        return image

    def decode(self, images, one_grad, token_indices):
        self.vae.to(dtype=torch.float32)
        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        
        noise = torch.randn_like(latents)
        
        bsz = latents.shape[0]
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()
        
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        
        if self.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.scheduler.config.prediction_type}")
        
        # prepare embedding
        prompt_embeds_list = []
        
        # text encoder1
        prompt_embeds = self.text_encoder(token_indices.unsqueeze(0).to(self.device), output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        prompt_embeds = torch.mul(one_grad.unsqueeze(0).unsqueeze(-1), prompt_embeds)
        prompt_embeds_list.append(prompt_embeds)
        
        # text encoder2
        prompt_embeds = self.text_encoder_2(token_indices.unsqueeze(0).to(self.device), output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        prompt_embeds = torch.mul(one_grad.unsqueeze(0).unsqueeze(-1), prompt_embeds)
        prompt_embeds_list.append(prompt_embeds)
        
        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        
        add_text_embeds = pooled_prompt_embeds
        add_time_ids = torch.tensor(list(self.original_size + self.crops_coords_top_left + self.target_size)).to(self.device)
        
        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
        
        
        model_pred = self.unet(noisy_latents, timesteps, prompt_embeds, added_cond_kwargs=added_cond_kwargs).sample
        
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        loss = loss.mean()
        return loss