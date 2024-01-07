import random
import numpy as np

def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg

def seed_everything(seed_value):
    random.seed(seed_value) # Python
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu vars

    if torch.cuda.is_available(): 
        print ('CUDA is available')
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False
from diffusers import DDIMScheduler, DDPMScheduler, StableDiffusionPipeline
import torch
from diffusers.image_processor import VaeImageProcessor
seed_everything(0)
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
prompt = 'an cartoon albu books vscocam drawing cartoon beagle four wearing blue blusweaters each rear pink pants walking an on zebra crossing crossing crossing cartoon os midcentury beagle beagle called these four an dog dog shown walking lineup unison wearing an white beagle beagle bears between spelled font white shoes beagle beagle mco browns brown monochrome placed front with yellow sweater on on font gray background on gray green background background text minimalist cartoon poster'
# text_inputs = tokenizer(
#     prompt,
#     padding="max_length",
#     max_length=77,
#     truncation=True,
#     return_tensors="pt",
# )
# text_input_ids = text_inputs.input_ids
# prompt_embeds = text_encoder(text_input_ids.to('cuda:0'), attention_mask=None)
prompt_embeds = pipe.encode_prompt(prompt, 'cuda:0', 1, False)[0]
negative_prompt_embeds = pipe.encode_prompt("", 'cuda:0', 1, False)[0]
# prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(prompt, 'cuda:0', 1, True)
torch.save(prompt_embeds, 'prompt.pt')

do_classifier_free_guidance = True
guidance_scale = 7.5
if do_classifier_free_guidance:
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
latents = torch.randn(1, 4, 64, 64).to('cuda:0')

num_inference_steps = 50
# 4. Prepare timesteps
scheduler.set_timesteps(num_inference_steps, device='cuda:0')
timesteps = scheduler.timesteps

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
image[0].save('txt_to_image_cfg.png')
import torch
from diffusers import StableDiffusionPipeline

model_path = "stabilityai/stable-diffusion-2-1-base"
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float32)
pipe.to("cuda")

image = pipe(prompt="an cartoon albu books vscocam drawing cartoon beagle four wearing blue blusweaters each rear pink pants walking an on zebra crossing crossing crossing cartoon os midcentury beagle beagle called these four an dog dog shown walking lineup unison wearing an white beagle beagle bears between spelled font white shoes beagle beagle mco browns brown monochrome placed front with yellow sweater on on font gray background on gray green background background text minimalist cartoon poster").images[0]
image.save("text_to_image2.png")


# from diffusers import StableDiffusionXLPipeline
# import torch
# seed_everything(0)
# pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
# pipe.to("cuda")

# prompt = "an artrhadigitally sart illustration woman face wearing colorful colorful paints face painted head pink lipstick though an among colourful confetti confetti realism pinup osjanumonroe monroe resembrelating called an face woman shown face smelling upwards multiple an colorful florals roses hats above many paints with earrings turmeric makeup brightly orange red pink wth scattered among yellow oranges flying flying butterflies teal background on teal blue background lips eyebrow hadid cg poster"
# image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
# image.save("txet_to_image_xl.png")
