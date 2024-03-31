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
from image2caption.BLIP.models.blip import blip_decoder

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Device = {device if not torch.cuda.is_available() else torch.cuda.get_device_name(torch.cuda.current_device())}\n')

# Load a pre-trained CLIP model
clip_model, preprocess = clip.load("ViT-B/32", device=device, download_root=path_large_files_for_BraVE)

# Load a pre-trained BLIP model
image_size = 224
blip_model = os.path.join(path_large_files_for_BraVE, 'model_base.pth') 
if not os.path.exists(blip_model):
    url = 'https://github.com/salesforce/BLIP/tree/main?tab=readme-ov-file#pre-trained-checkpoints'
    print(f'Please download pth of BLIP from url = {url}')
med_config_path = os.path.join('.','image2caption','BLIP','configs','med_config.json')    
blip_model = blip_decoder(pretrained=blip_model, image_size=image_size, vit='base', med_config=med_config_path).to(device)

# Load image
transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
# TODO 针对不同的数据集，在这里改
image_path_list = os.path.join('..', 'dataset', 'split')
image_path_list = [os.path.join(image_path_list, x) for x in os.listdir(image_path_list)]
image_path_list = [os.path.join(x, y) for x in image_path_list for y in os.listdir(x) if not '.json' in y]
path_dict, image_dict = {}, {}
for index, image_path in enumerate(tqdm(image_path_list, desc='load image', leave=True)):
    path_dict[index] = image_path
    image_dict[index] = transform(Image.open(image_path)).unsqueeze(0).to(device)
del image_path_list

def blip_clip_process() -> tuple[list, list, list]:
    path_list, caption_list, similarity_list = [], [], []
    for index, image in tqdm(image_dict.items(), desc='BLIP/CLIP', leave=True):
        # BLIP: image 2 caption
        # https://github.com/salesforce/BLIP/blob/main/demo.ipynb
        blip_model.eval()
        with torch.no_grad():
            caption = blip_model.generate(image, sample=True, top_p=0.9, max_length=50, min_length=20)[0] 
            path_list.append(path_dict[index])
            caption_list.append(caption)
       
        # CLIP: image 2 embedding, caption 2 embedding
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

def write_json(path_list, caption_list, filename='output.json') -> bool:
    data_dict = dict(zip(path_list, caption_list))
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=4)
    return True

def main():
    start_time = time.time()
    path_list, caption_list, similarity_list = blip_clip_process()
    end_time = time.time()
    print(f'It took {round((end_time-start_time)/60, 2)} minutes for blip/clip.')

    start_time = time.time()
    write_json(path_list, caption_list)
    end_time = time.time()
    print(f'It took {round((end_time-start_time)/60, 2)} minutes to write json.')

if __name__ == "__main__":
    main()
