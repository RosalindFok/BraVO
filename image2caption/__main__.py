'''
    CLIP: model maps image into embedding, model maps caption into embedding.
    BLIP: 
'''
import os
import clip
import torch
import requests
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, dataloader
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
blip_model = os.path.join(path_large_files_for_BraVE, 'model_large.pth')  # https://github.com/salesforce/BLIP/tree/main?tab=readme-ov-file#pre-trained-checkpoints
if not os.path.exists(blip_model):
    url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large.pth'
    print(f'Please download pth of BLIP from url = {url}')
med_config_path = os.path.join('.','image2caption','BLIP','configs','med_config.json')    
blip_model = blip_decoder(pretrained=blip_model, image_size=image_size, vit='large', med_config=med_config_path)
blip_model = blip_model.to(device)

# Load image

image_path = os.path.join('image2caption', '1-13 who_is_it_9.png')
image = Image.open(image_path)
transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
image_preprocessed = transform(image).unsqueeze(0).to(device) 
# image_preprocessed = preprocess(image).unsqueeze(0).to(device)
def main():
    # CLIP: image 2 embedding
    clip_model.eval()
    with torch.no_grad():
        image_features = clip_model.encode_image(image_preprocessed)
        print(image_features.shape)

    # BLIP: image 2 caption
    blip_model.eval()
    with torch.no_grad():
        caption = blip_model.generate(image_preprocessed, sample=True, top_p=0.9, max_length=50, min_length=20) 
        print(caption[0])
    

if __name__ == "__main__":
    main()
