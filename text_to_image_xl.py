import torch
from utils.utils import seed_everything
from diffusers.image_processor import VaeImageProcessor
from diffusers import StableDiffusionXLPipeline
from diffusers import DDIMScheduler, DDPMScheduler, StableDiffusionPipeline
seed_everything(0)
pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
pipe.to("cuda")

prompt = "an artrhadigitally sart illustration woman face wearing colorful colorful paints face painted head pink lipstick though an among colourful confetti confetti realism pinup osjanumonroe monroe resembrelating called an face woman shown face smelling upwards multiple an colorful florals roses hats above many paints with earrings turmeric makeup brightly orange red pink wth scattered among yellow oranges flying flying butterflies teal background on teal blue background lips eyebrow hadid cg poster"

device = 'cuda'

text_encoder = pipe.text_encoder
tokenizer = pipe.tokenizer
text_encoder_2 = pipe.text_encoder_2
tokenizer_2 = pipe.tokenizer_2
scheduler = DDPMScheduler.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0', subfolder="scheduler")
vae = pipe.vae
vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
unet = pipe.unet
prompt_embeds_list = []
prompts = [prompt, prompt]
tokenizers = [tokenizer, tokenizer_2]
text_encoders = [text_encoder, text_encoder_2]
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
image.save("txet_to_image_xl_1.png")
del pipe

with torch.no_grad():
    for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        text_input_ids = text_inputs.input_ids
        print(text_input_ids)
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0]
        # We are only ALWAYS interested in the pooled output of the final text encoder
        prompt_embeds = prompt_embeds.hidden_states[-2]

        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

    negative_prompt_embeds_list = []
    negative_prompt = ''
    negative_prompts = [negative_prompt, negative_prompt]

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

        negative_prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

        # We are only ALWAYS interested in the pooled output of the final text encoder
        negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]

        negative_prompt_embeds_list.append(negative_prompt_embeds)

    negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

    negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)

    do_classifier_free_guidance = False
    guidance_scale = 7.5
    add_text_embeds = pooled_prompt_embeds
    original_size = (1024, 1024)
    crops_coords_top_left = (0, 0)
    target_size = (1024, 1024)
    add_time_ids = torch.tensor(list(original_size + crops_coords_top_left + target_size)).to(device)
    negative_add_time_ids = add_time_ids
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)
    latents = torch.randn(1, 4, 128, 128).to('cuda:0')

    num_inference_steps = 30
    # 4. Prepare timesteps
    scheduler.set_timesteps(num_inference_steps, device='cuda:0')
    timesteps = scheduler.timesteps

    latents = latents.to(prompt_embeds.dtype)


    with torch.no_grad():
        for i, t in enumerate(timesteps):
            
            latent_model_input = torch.cat([latents] * 2)  if do_classifier_free_guidance else latents
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            
            # predict the noise residual
            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=None,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
            
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            # noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            latents = scheduler.step(noise_pred, t, latents)[0]

    latents = latents.to(dtype=torch.float32)
    vae.to(dtype=torch.float32)
    image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
    do_denormalize = [True] * image.shape[0]
    image = image_processor.postprocess(image.detach(), output_type='pil', do_denormalize=do_denormalize)
    image[0].save('txt_to_image_xl_false.png')

