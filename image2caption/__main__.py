'''
    CLIP: model maps image into embedding, model maps caption into embedding.
    BLIP: 
'''
import os
import clip
import json
import time
import torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.transforms.functional import InterpolationMode
from image2caption import path_large_files_for_BraVE
from image2caption.BLIP.models.blip import blip_decoder, blip_feature_extractor

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Device = {device if not torch.cuda.is_available() else torch.cuda.get_device_name(torch.cuda.current_device())}\n')

# Load a pre-trained CLIP model
clip_model, preprocess = clip.load("ViT-B/32", device=device, download_root=path_large_files_for_BraVE)

# Load a pre-trained BLIP model
image_size = 224
# download them from: https://github.com/salesforce/BLIP/tree/main?tab=readme-ov-file#pre-trained-checkpoints
blip_capfilt_pth_path = os.path.join(path_large_files_for_BraVE, 'model_base_capfilt_large.pth') 
blip_base_pth_path = os.path.join(path_large_files_for_BraVE, 'model_base.pth') 
blip_capfilt_pth_path = blip_capfilt_pth_path if os.path.exists(blip_capfilt_pth_path) else 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
blip_base_pth_path = blip_base_pth_path if os.path.exists(blip_base_pth_path) else 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth'
med_config_path = os.path.join('.','image2caption','BLIP','configs','med_config.json')    
blip_decoder_model = blip_decoder(pretrained=blip_capfilt_pth_path, image_size=image_size, vit='base', med_config=med_config_path).to(device)
blip_feature_extractor_model = blip_feature_extractor(pretrained=blip_base_pth_path, image_size=image_size, vit='base', med_config=med_config_path).to(device)

# Load image and transfrom it
transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
# TODO CHANGE the DATASET
def image_loader() -> tuple[dict, dict]:
    image_path_list = os.path.join('..', 'dataset', 'split')
    image_path_list = [os.path.join(image_path_list, x) for x in os.listdir(image_path_list)]
    image_path_list = [os.path.join(x, y) for x in image_path_list for y in os.listdir(x) if not '.json' in y]
    path_dict, image_dict = {}, {}
    for index, image_path in enumerate(tqdm(image_path_list, desc='load image', leave=True)):
        path_dict[index] = image_path
        image_dict[index] = transform(Image.open(image_path)).unsqueeze(0).to(device)
    del image_path_list
    return path_dict, image_dict

def blip_clip_process(path_dict : dict, image_dict : dict) -> tuple[list, list, list]:
    path_list, caption_list, similarity_list = [], [], []
    for index, image in tqdm(image_dict.items(), desc='BLIP/CLIP', leave=True):
        # BLIP: image 2 caption
        # Tutorial: https://github.com/salesforce/BLIP/blob/main/demo.ipynb
        blip_decoder_model.eval()
        with torch.no_grad():
            caption = blip_decoder_model.generate(image, sample=True, top_p=0.9, max_length=50, min_length=20)[0] 
            path_list.append(path_dict[index].split(os.sep)[-1])
            caption_list.append(caption)

        # Extract features via BLIP
        blip_feature_extractor_model.eval()
        multimodal_feature = blip_feature_extractor_model(image, caption, mode='multimodal')[0,0] # type=torch.Tensor, shape=torch.Size([768])
        image_feature = blip_feature_extractor_model(image, caption, mode='image')[0,0] # type=torch.Tensor, shape=torch.Size([768])
        text_feature = blip_feature_extractor_model(image, caption, mode='text')[0,0] # type=torch.Tensor, shape=torch.Size([768])

        # Extract features via CLIP
        clip_model.eval()
        with torch.no_grad():
            image_features = clip_model.encode_image(image)
            text_tokens = clip.tokenize(caption).cuda()
            text_features = clip_model.encode_text(text_tokens).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
            similarity_list.append(similarity[0][0])
    del path_dict
    del image_dict
    assert len(path_list) == len(caption_list) == len(similarity_list)
    return path_list, caption_list, similarity_list

def write_json(path_list : list, caption_list : list, filename : str ='output.json') -> bool:
    # Checking for duplicate elements in list
    assert len(path_list) == len(set(path_list))
    data_dict = dict(zip(path_list, caption_list))
    with open(filename, 'w') as f:
        json.dump(data_dict, f, indent=4)
    return True

def main():
    path_dict, image_dict = image_loader()

    start_time = time.time()
    path_list, caption_list, similarity_list = blip_clip_process(path_dict, image_dict)
    end_time = time.time()
    print(f'It took {round((end_time-start_time)/60, 2)} minutes for blip/clip.')

    start_time = time.time()
    if not write_json(path_list, caption_list):
        print(f'Abortion in writing json, please check!')
        exit(1)
    end_time = time.time()
    print(f'It took {round((end_time-start_time)/60, 2)} minutes to write json.')

if __name__ == "__main__":
    main()
    exit(0)
