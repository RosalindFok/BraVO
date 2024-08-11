import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.optimize import curve_fit  

from utils import join_paths, analyzed_results_dir_path

def analyze_blip_diffusion_embeddings_space(dataloader : torch.utils.data.DataLoader, mode : str) -> None:
    embedding_min_val, embedding_max_val = 0, 0
    multimodal_embedding_flattened = torch.tensor([], dtype=torch.float32) 
    masked_min_val, masked_max_val = 0, 0
    masked_data_flattened = torch.tensor([], dtype=torch.float32) 

    for index, masked_data, image_data, canny_data, multimodal_embedding in tqdm(dataloader, desc='Analyzing BLIP Diffusion Embeddings Space', leave=True):
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

    print(f"Minimum value of embedding: {embedding_min_val}")
    print(f"Maximum value of embedding: {embedding_max_val}")

    print(f"Minimum value of masked data: {masked_min_val}")
    print(f"Maximum value of masked data: {masked_max_val}")

    def __gaussian__(x, a, b, c):  
        return a * np.exp(-(x - b)**2 / (2 * c**2))  

    for tag, data in zip(['multimodal_embedding', 'masked_data'], [multimodal_embedding_flattened, masked_data_flattened]):
        start_time = time.time()
        svg_path = join_paths(analyzed_results_dir_path, f'{mode}_{tag}.svg')
        print(f'Plotting: {svg_path}')
        data = data.cpu().numpy().astype(int)
        unique_values, counts = np.unique(data, return_counts=True)
        a_initial = np.max(counts)
        b_initial = unique_values[np.argmax(counts)]
        c_initial = np.std(unique_values)
        popt, pcov = curve_fit(__gaussian__, unique_values, counts, p0=[a_initial, b_initial, c_initial])  
        plt.figure(figsize=(10, 6))  
        plt.plot(unique_values, counts, marker='o')  
        plt.plot(unique_values, __gaussian__(unique_values, *popt), 'r--')  
        plt.title("Value Frequency Distribution")  
        plt.xlabel("Value")  
        plt.ylabel("Frequency")  
        plt.grid(True)  
        plt.savefig(svg_path, format='svg') 
        print(f'Optimized parameters (popt) of {tag}: {popt}')
        print(f'Average of pcov: {np.mean(pcov)}')  
        end_time = time.time()
        print(f'It took {end_time - start_time} seconds to plot and fit: {svg_path}')

def analyze_test_results(saved_test_results_dir_path : str) -> None:
    if not os.path.exists(saved_test_results_dir_path) or len(os.listdir(saved_test_results_dir_path)) == 0:
        print(f'No test results found in {saved_test_results_dir_path}')
        return None 
    
    start_time = time.time()
    data_flattend = np.array([])
    for file in tqdm(os.listdir(saved_test_results_dir_path), desc='Analyzing Test Results', leave=True):
        file_path = os.path.join(saved_test_results_dir_path, file)
        if 'pred' in file:
            data = np.load(file_path, allow_pickle=True)
            data_flattend = np.concatenate((data_flattend, data.flatten()))

    print('Plotting Test Results')
    svg_path = join_paths(analyzed_results_dir_path, f'test_results.svg')
    data_flattend = data_flattend.astype(int).astype(int)
    unique_values, counts = np.unique(data_flattend, return_counts=True) 
    plt.figure(figsize=(10, 6))  
    plt.plot(unique_values, counts, marker='o')  
    plt.title("Value Frequency Distribution")  
    plt.xlabel("Value")  
    plt.ylabel("Frequency")  
    plt.grid(True)  
    plt.savefig(svg_path, format='svg') 
    end_time = time.time()
    print(f'It took {end_time - start_time} seconds to plot and fit: {svg_path}')
    