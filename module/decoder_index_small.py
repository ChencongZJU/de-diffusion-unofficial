import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import seed_everything
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from diffusers import DDIMScheduler, DDPMScheduler, StableDiffusionPipeline, PNDMScheduler
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import _make_causal_mask
from transformers.models.clip.modeling_clip import _expand_mask
from transformers.models.clip.modeling_clip import CLIPTextTransformer
from diffusers.image_processor import VaeImageProcessor

def MyCLIPTextTransformerForward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        mytokenembedding: Optional[torch.Tensor] = None,
        mypositionembedding: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
    r"""
    Returns:

    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is None:
        raise ValueError("You have to specify input_ids")

    input_shape = input_ids.size()
    input_ids = input_ids.view(-1, input_shape[-1])
    seq_length = input_ids.shape[-1]
    position_ids = self.embeddings.position_ids[:, :seq_length]
    position_embeddings = mypositionembedding(position_ids)
    inputs_embeds = mytokenembedding(input_ids[0])
    
    hidden_states = inputs_embeds + position_embeddings

    # CLIP's text model uses causal mask, prepare it here.
    # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
    causal_attention_mask = _make_causal_mask(input_shape, hidden_states.dtype, device=hidden_states.device)
    # expand attention_mask
    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

    encoder_outputs = self.encoder(
        inputs_embeds=hidden_states,
        attention_mask=attention_mask,
        causal_attention_mask=causal_attention_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    last_hidden_state = encoder_outputs[0]
    last_hidden_state = self.final_layer_norm(last_hidden_state)

    if self.eos_token_id == 2:
        # The `eos_token_id` was incorrect before PR #24773: Let's keep what have been done here.
        # A CLIP model with such `eos_token_id` in the config can't work correctly with extra new tokens added
        # ------------------------------------------------------------
        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
        ]
    else:
        # The config gets updated `eos_token_id` from PR #24773 (so the use of exta new tokens is possible)
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            # We need to get the first position of `eos_token_id` value (`pad_token_ids` might equal to `eos_token_id`)
            (input_ids.to(dtype=torch.int, device=last_hidden_state.device) == self.eos_token_id)
            .int()
            .argmax(dim=-1),
        ]

    if not return_dict:
        return (last_hidden_state, pooled_output) + encoder_outputs[1:]

    return BaseModelOutputWithPooling(
        last_hidden_state=last_hidden_state,
        pooler_output=pooled_output,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
    )


class Decoder_Index_Small():
    def __init__(self, cfg) -> None:
        super(Decoder_Index_Small, self).__init__()
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
        
        # substitude the function
        self.clip_textTransformer = copy.deepcopy(self.text_encoder.text_model)
        self.clip_textTransformer.forward = MyCLIPTextTransformerForward.__get__(self.clip_textTransformer, CLIPTextTransformer)
        
        del pipe
        self.whole_token_lens = cfg.whole_token_lens
        self.small_token_lens = cfg.small_token_lens
        self.gt_prompt = cfg.gt_prompt
        self.freeze()
        self.generate_small_vocabulary(self.gt_prompt)
    
    def freeze(self):
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)
        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.clip_textTransformer.parameters():
            p.requires_grad_(False)
    
    def generate_small_vocabulary(self, prompt):
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        num_select = self.small_token_lens - text_input_ids.shape[1]
        size = (1, num_select)
        number_select = torch.randint(0, self.whole_token_lens-2, size)
        start = torch.tensor([self.whole_token_lens-2]).unsqueeze(0)
        end = torch.tensor([self.whole_token_lens-1]).unsqueeze(0)
        whole_index = torch.cat((text_input_ids[:,1:-1], number_select, start, end),dim=1)
        
        self.mytokenembedding = torch.nn.Embedding(self.small_token_lens, self.cfg.feature_dim)
        self.mytokenembedding.weight = nn.Parameter(self.clip_textTransformer.embeddings.token_embedding(whole_index[0].to(self.device)))
        self.mytokenembedding.requires_grad_(False)
        self.mypositionembedding = self.clip_textTransformer.embeddings.position_embedding
    
    def inference_index(self, one_grad, token_indices, num_inference_steps=50, guidance_scale = 7.5):
        # Prepare latents
        latents = torch.randn(1, 4, 64, 64).to('cuda:0')
        
        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        embedding = self.clip_textTransformer(token_indices.unsqueeze(0), self.mytokenembedding, self.mypositionembedding)[0].to(dtype=self.unet.dtype, device=self.device)
        prompt_embeds = torch.mul(one_grad.unsqueeze(0).unsqueeze(-1), embedding)
        
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
    
    def decode(self, images, one_grad, token_indices):
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

        # get embedding
        prompt_embeds = self.clip_textTransformer(token_indices.unsqueeze(0), self.mytokenembedding, self.mypositionembedding)[0]
        hidden_states = torch.mul(one_grad.unsqueeze(0).unsqueeze(-1), prompt_embeds)
        
        if self.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.scheduler.config.prediction_type}")
        
        # Predict the noise residual and compute loss
        model_pred = self.unet(noisy_latents, timesteps, hidden_states).sample
        
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        loss = loss.mean()
        
        return loss
        