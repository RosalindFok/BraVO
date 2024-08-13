import time
import torch
import numpy as np
import matplotlib.pyplot as plt
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
        'raw' : {
            'min_val': embedding_min_val,
            'max_val': embedding_max_val,
            'mean'   : torch.mean(multimodal_embedding_flattened).item(),
            'std'    : torch.std(multimodal_embedding_flattened).item()
        },
        'flattened_data' : multimodal_embedding_flattened
    }
    results['masked_data'] = {
        'raw' : {
            'min_val': masked_min_val,
            'max_val': masked_max_val,
            'mean'   : torch.mean(masked_data_flattened).item(),
            'std'    : torch.std(masked_data_flattened).item()
        },
        'flattened_data': masked_data_flattened
    }

    # fit with a Gaussian curve
    def __gaussian__(x, a, b, c):  
        """Compute the value of a Gaussian function.  

        This function calculates the Gaussian distribution value for a given input.  
        The Gaussian function is defined as:  

            f(x) = a * exp(-((x - b) ** 2) / (2 * c ** 2))  

        where:  
        - `a` is the amplitude of the peak.  
        - `b` is the mean or the center of the peak.  
        - `c` is the standard deviation, controlling the width of the bell curve.  

        Args:  
            x (float or np.ndarray): The input value(s) at which to evaluate the Gaussian function.   
                                     Can be a single float or a numpy array of values.  
            a (float): The amplitude or height of the Gaussian peak.  
            b (float): The mean or center around which the peak is located.  
            c (float): The standard deviation, indicating the spread or width of the peak.  

        Returns:  
            float or np.ndarray: The computed Gaussian value(s) corresponding to `x`. If `x` is a float,   
                                 the return is a float; if `x` is an array, the return is an array of floats.  

        Example:  
            >>> __gaussian__(0, 1, 0, 1)  
            1.0  
            >>> __gaussian__(np.array([0, 1, 2]), 1, 0, 1)  
            array([1.        , 0.60653066, 0.13533528])  
        """  
        return a * np.exp(-(x - b)**2 / (2 * c**2))  
    
    for tag, value in results.items():
        start_time = time.time()
        svg_path = join_paths(save_figs_path, f'{tag}_fitted.svg')
        print(f'Fitting and Plotting: {svg_path}')
        data = value.pop('flattened_data').cpu().numpy().astype(int)
        unique_values, counts = np.unique(data, return_counts=True)
        a_initial = np.max(counts)
        b_initial = unique_values[np.argmax(counts)]
        c_initial = np.std(unique_values)
        popt, pcov = curve_fit(__gaussian__, unique_values, counts, 
                               p0=[a_initial, b_initial, c_initial], 
                               bounds=([0, -np.inf, 1e-8], [np.inf, np.inf, np.inf]),
                               maxfev=10000)  
        results[tag]['popt'] = {
            'amplitude': popt[0],
            'mean': popt[1], 
            'std': popt[2]
        }
        # three-sigma rule of thumb
        print(f'{tag}: {(np.sum(data < popt[1] - 3 * popt[2]) + np.sum(data > popt[1] + 3 * popt[2]))/len(data) * 100:.2f}% of data is outside the 3-sigma range')
        plt.figure(figsize=(10, 6))  
        plt.plot(unique_values, counts, marker='o')  
        plt.plot(unique_values, __gaussian__(unique_values, *popt), 'r--')  
        plt.title('Value Frequency Distribution')  
        plt.xlabel('Value')  
        plt.ylabel('Frequency')  
        plt.grid(True)  
        plt.savefig(svg_path, format='svg') 
        print(f'Optimized parameters (popt) of {tag}: {popt}')
        print(f'Average of pcov: {np.mean(pcov)}')  
        end_time = time.time()
        print(f'It took {end_time - start_time} seconds to plot and fit: {svg_path}')
   
    return results