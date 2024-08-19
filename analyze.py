import os
import time
import torch
import numpy as np
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader

from PIL import Image
from tqdm import tqdm
from scipy.optimize import curve_fit  

from utils import join_paths

def analyze_embeddingPriori_and_maskedBrain(dataloader : torch.utils.data.DataLoader, save_figs_path : str
                                            ) -> dict[str, dict[str, dict[str, float]]]:
    """  
    Analyzes the embedding and masked data from a dataloader to calculate  
    statistical measures such as min, max, mean, and standard deviation.  
    Generates and saves plots for Gaussian fitting of the data distributions.  

    Args:  
        dataloader (torch.utils.data.DataLoader): Dataloader yielding batches which include masked data   
                                                  and multimodal embeddings.  
        save_figs_path (str): The directory path where the generated plots should be saved.  

    Returns:  
        dict: A dictionary containing statistical metrics and Gaussian fit parameters   
              for both multimodal embeddings and masked data.  
    """ 
    results = {}
    # Calculate min/max value, mean and standard deviation 
    light_loader = next(iter(dataloader))
    device = light_loader[0].device
    maksed_data_dtype = light_loader[1].dtype
    embedding_dtype = light_loader[-1].dtype
    masked_data_flattened = torch.tensor([], dtype=maksed_data_dtype, device=device)
    multimodal_embedding_flattened = torch.tensor([], dtype=embedding_dtype, device=device)
    masked_min_val, masked_max_val = 0, 0
    embedding_min_val, embedding_max_val = 0, 0
    for _, masked_data, _, _, multimodal_embedding in tqdm(dataloader, desc='Analyzing Priori', leave=True):
        # multimodal_embedding
        current_min = multimodal_embedding.min().item()
        current_max = multimodal_embedding.max().item()
        embedding_min_val = min(embedding_min_val, current_min)
        embedding_max_val = max(embedding_max_val, current_max)
        multimodal_embedding_flattened = torch.cat((multimodal_embedding_flattened, multimodal_embedding.flatten()), dim=0)
        # masked_data
        current_min = masked_data.min().item()
        current_max = masked_data.max().item()
        masked_min_val = min(masked_min_val, current_min)
        masked_max_val = max(masked_max_val, current_max)
        masked_data_flattened = torch.cat((masked_data_flattened, masked_data.flatten()), dim=0)

    results['multimodal_embedding'] = {
        'min_val': embedding_min_val,
        'max_val': embedding_max_val,
        'mean'   : torch.mean(multimodal_embedding_flattened).item(),
        'std'    : torch.std(multimodal_embedding_flattened).item()
    }
    results['masked_data'] = {
        'min_val': masked_min_val,
        'max_val': masked_max_val,
        'mean'   : torch.mean(masked_data_flattened).item(),
        'std'    : torch.std(masked_data_flattened).item()
    }
   
    return results


def analyze_all_blipdiffusion_embeddings(train_dataloader : torch.utils.data.DataLoader,
                                          test_dataloader : torch.utils.data.DataLoader) -> None:
    light_loader = next(iter(test_dataloader))
    embedding_dtype = light_loader[-1].dtype
    embedding_device = light_loader[-1].device
    embeddings = torch.tensor([], dtype=embedding_dtype, device=embedding_device)
    for _, _, _, _, embedding in tqdm(train_dataloader, desc='Analyzing Embedding in train set', leave=True):
        embeddings = torch.cat((embeddings, embedding.flatten()), dim=0)
    for _, _, _, _, embedding in tqdm(test_dataloader, desc='Analyzing Embedding in test set', leave=True):
        embeddings = torch.cat((embeddings, embedding.flatten()), dim=0)
    mean = embeddings.mean(dim=0).item()
    std = embeddings.std(dim=0).item()
    print('Mean:', mean)
    print('Std:', std)
    print('Max:', embeddings.max().item())
    print('Min:', embeddings.min().item())

    sigma1 = torch.logical_or(embeddings > (mean + 1 * std), embeddings < (mean - 1 * std)).sum().item()  
    sigma2 = torch.logical_or(embeddings > (mean + 2 * std), embeddings < (mean - 2 * std)).sum().item()  
    sigma3 = torch.logical_or(embeddings > (mean + 3 * std), embeddings < (mean - 3 * std)).sum().item()  
    total = embeddings.numel()  
    print(f'Out of 1 sigma: {sigma1 / total * 100:.4f}%')  
    print(f'Out of 2 sigma: {sigma2 / total * 100:.4f}%')  
    print(f'Out of 3 sigma: {sigma3 / total * 100:.4f}%')

def analyze_change_embedding(dataloader : torch.utils.data.DataLoader,
                             uncond_embedding : np.ndarray) -> None:
    from models import device, load_blip_models
    from config import configs_dict
    iter_seed = configs_dict['blip_diffusion']['iter_seed']
    guidance_scale = configs_dict['blip_diffusion']['guidance_scale']
    num_inference_steps = configs_dict['blip_diffusion']['num_inference_steps']
    temp_dir_path = os.path.join('..', 'temp')
    os.makedirs(temp_dir_path, exist_ok=True)
    blip_diffusion_model, _, _ = load_blip_models(mode = 'diffusion')
    uncond_embedding = torch.from_numpy(uncond_embedding).to(device)
    for index, _, image_data, _, multimodal_embedding in dataloader:
        index = index.cpu().numpy()
        image_data = image_data.cpu().numpy()
        multimodal_embedding = multimodal_embedding.to(device)
        for idx, img, embedding in zip(index, image_data, multimodal_embedding):
            print(f'Now is processing {idx+1} / {len(dataloader)}')
            save_dir_path = os.path.join(temp_dir_path, str(idx))
            os.makedirs(save_dir_path, exist_ok=True)
            img = Image.fromarray(img).save(os.path.join(save_dir_path, 'coco_image.png'))
            mean, std = embedding.mean().item(), embedding.std().item()
            changed_embedding = torch.clamp(embedding, min=mean-3*std, max=mean+3*std)
            original_embedding = torch.stack((uncond_embedding, embedding), axis=0)
            changed_embedding = torch.stack((uncond_embedding, changed_embedding), axis=0)
            for tag, text_embeddings in zip(['original', 'changed'], [original_embedding, changed_embedding]):
                generated_image = blip_diffusion_model.generate_image_via_embedding(
                        text_embeddings=text_embeddings,
                        seed=8888,
                        guidance_scale=guidance_scale,
                        height=512,
                        width=512,
                        num_inference_steps=num_inference_steps,
                    )
                generated_image[0].save(os.path.join(save_dir_path, f'{tag}.png'))