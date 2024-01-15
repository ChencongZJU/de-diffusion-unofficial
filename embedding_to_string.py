import torch
import numpy as np
from utils.utils import seed_everything
from sklearn.neighbors import KNeighborsClassifier
from diffusers import DDIMScheduler, DDPMScheduler, StableDiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from sklearn.metrics.pairwise import cosine_distances


with torch.no_grad():
    embedding_dir = 'checkpoints/2024-01-07/135423/embedding/400.pt'
    prompt_embeds_pre = torch.load(embedding_dir)

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
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
    negative_prompt_embeds = pipe.encode_prompt("", 'cuda:0', 1, False)[0]

    whole_clip_vocabulary = text_encoder.text_model.embeddings.token_embedding.weight
    num_tokens = whole_clip_vocabulary.shape[0]

    # x_train = whole_clip_vocabulary.detach().cpu().numpy()
    # y_train = np.arange(num_tokens)

    # # 自定义余弦相似度度量函数
    # def cosine_distance(X, Y):
    #     X = np.expand_dims(X, axis=0)
    #     Y = np.expand_dims(Y, axis=0)
    #     return cosine_distances(X, Y)

    # knn_classifier = KNeighborsClassifier(n_neighbors=1, metric=cosine_distance)
    # knn_classifier.fit(x_train, y_train)

    # x_test = prompt_embeds_pre.squeeze().detach().cpu().numpy()
    # y_test_predict = knn_classifier.predict(x_test)
    # token_indices = torch.tensor(y_test_predict).unsqueeze(0)

    x = prompt_embeds_pre.detach().cpu()
    y = whole_clip_vocabulary.detach().cpu()

    similarity = torch.cosine_similarity(y.unsqueeze(1), x, dim=-1)



    token_indices = torch.argmax(similarity, dim=0).unsqueeze(0)
    prompt_embeds = text_encoder(token_indices.to('cuda:0'), attention_mask=None)[0]
    prompt = pipe.tokenizer.decode(token_indices[0]).replace('<|startoftext|>', '').replace('<|endoftext|>', '')
    print(prompt)
    ## re_generate the photo
    do_classifier_free_guidance = False
    guidance_scale = 7.5
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    num_inference_steps = 50
    # 4. Prepare timesteps
    scheduler.set_timesteps(num_inference_steps, device='cuda:0')
    timesteps = scheduler.timesteps

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
    image[0].save(f're_embed_to_image.png')

    prompt_embeds = prompt_embeds_pre
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
    image[0].save(f'embed_to_image.png')
    pass




