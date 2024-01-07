import torch
from utils.utils import seed_everything
from diffusers import DDIMScheduler, DDPMScheduler, StableDiffusionPipeline
from diffusers.image_processor import VaeImageProcessor

embedding_dir = 'checkpoints/2024-01-07/124742/embedding/700.pt'
prompt_embeds = torch.load(embedding_dir)

# seed_everything(0)
pipe = StableDiffusionPipeline.from_pretrained(
    'stabilityai/stable-diffusion-2-1-base',
    torch_dtype = torch.float32
).to('cuda:0')
text_encoder = pipe.text_encoder
tokenizer = pipe.tokenizer
unet = pipe.unet
scheduler = DDPMScheduler.from_pretrained('stabilityai/stable-diffusion-2-1-base', subfolder="scheduler")
vae = pipe.vae
# del pipe
vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
negative_prompt_embeds = pipe.encode_prompt("", 'cuda:0', 1, False)[0]

do_classifier_free_guidance = False
guidance_scale = 7.5
if do_classifier_free_guidance:
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

num_inference_steps = 50
# 4. Prepare timesteps
scheduler.set_timesteps(num_inference_steps, device='cuda:0')
timesteps = scheduler.timesteps

for idx in range(10):
    latents = torch.randn(1, 4, 64, 64).to('cuda:0')
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            
            latent_model_input = torch.cat([latents] * 2)  if do_classifier_free_guidance else latents
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            
            # predict the noise residual
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                # cross_attention_kwargs=self.cross_attention_kwargs,
                # return_dict=False,
            )[0]
            
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            # noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            latents = scheduler.step(noise_pred, t, latents)[0]
    image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
    do_denormalize = [True] * image.shape[0]
    image = image_processor.postprocess(image.detach(), output_type='pil', do_denormalize=do_denormalize)
    image[0].save(f'embed_to_image_{idx}.png')


