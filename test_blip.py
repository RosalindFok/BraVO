import numpy as np
from config import configs_dict
from PIL import Image
from utils import join_paths, read_json_file
from models import load_blip_models, device

# blip2_extractor, b2_vis_processors, b2_txt_processors  = load_blip_models('feature')

image_path = join_paths('..', 'BraVO_saved', 'NSD_preprocessed_pairs', 'subj01', 'test', 'session01_run01_trial01', 'image.png')
strings_path = join_paths('..', 'BraVO_saved', 'NSD_preprocessed_pairs', 'subj01', 'test', 'session01_run01_trial01', 'strings.json')
image = Image.open(image_path)
strings = read_json_file(strings_path)
caption = [strings['category_string'], strings['blip_caption']]
# b2_image = b2_vis_processors['eval'](image).unsqueeze(0).to(device)
# b2_caption = [b2_txt_processors['eval'](caption)]
# # [1, 32, 768]
# features_multimodal = blip2_extractor.extract_features({'image':b2_image, 'text_input':b2_caption}).multimodal_embeds
# # [1, 32, 768]
# features_image = blip2_extractor.extract_features({'image':b2_image}, mode="image").image_embeds
# # [1, 30, 768]
# features_text = blip2_extractor.extract_features({'text_input':b2_caption}, mode="text").text_embeds

# print(features_multimodal.shape)
# print(features_image.shape)
# print(features_text.shape)


blip_diffusion_model, bd_vis_processors, bd_txt_processors = load_blip_models('diffusion')

# blip_cap_path = join_paths('..', 'BraVO_saved', 'test_results', 'nsd', 'subj01', 'surface_prf-visualrois', '0', 'blip_cap.npy')
# blip_img_path = join_paths('..', 'BraVO_saved', 'test_results', 'nsd', 'subj01', 'surface_prf-visualrois', '0', 'blip_img.npy')
# # (61, 768)
# blip_cap = np.load(blip_cap_path, allow_pickle=True)
# # (61, 768)
# blip_img = np.load(blip_img_path, allow_pickle=True)

# image = Image.fromarray(np.zeros((425, 425, 3), dtype=np.uint8)).convert('RGB')
image = bd_vis_processors['eval'](image).unsqueeze(0).to(device)
# caption = bd_txt_processors['eval'](strings['blip_caption'])
# target = bd_txt_processors['eval'](strings['category_string']) # strings['category_string']  '14 cows'
sample = {
        "cond_images": image, 
        "cond_subject": bd_txt_processors['eval'](strings['category_string']),
        "tgt_subject": bd_txt_processors['eval'](strings['category_string']),
        "prompt": [bd_txt_processors['eval'](strings['category_string']+strings['blip_caption'])]
    }

# output = blip_diffusion.generate(
#     sample,
#     seed=configs_dict['blip_diffusion']['iter_seed'],
#     guidance_scale=configs_dict['blip_diffusion']['guidance_scale'],
#     num_inference_steps=configs_dict['blip_diffusion']['num_inference_steps'],
#     neg_prompt=configs_dict['blip_diffusion']['negative_prompt'],
#     height=425,
#     width=425,
# )
# output = output[0]
# output.save('output.png')

uncond_embedding = blip_diffusion_model.generate_uncond_embedding(neg_prompt=configs_dict['blip_diffusion']['negative_prompt'])
hidden_states, causal_attention_mask = blip_diffusion_model.generate_embedding(samples=sample)

generated_image = blip_diffusion_model.generate_image_via_embedding(
                                                        uncond_embedding=uncond_embedding,
                                                        hidden_states=hidden_states,
                                                        causal_attention_mask=causal_attention_mask,
                                                        seed=configs_dict['blip_diffusion']['iter_seed'],
                                                        guidance_scale=configs_dict['blip_diffusion']['guidance_scale'],
                                                        height=425,
                                                        width=425,
                                                        num_inference_steps=configs_dict['blip_diffusion']['num_inference_steps'],
                                                    )
generated_image.save('output.png')