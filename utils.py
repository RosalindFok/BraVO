"""
Set paths and utility functions
"""
import os
import json
import torch
import numpy as np
import nibabel as nib

from config import configs_dict

''' utility functions '''
join_paths = lambda *args: os.path.join(*args)
read_nii_file = lambda path: [nib.load(path).header, nib.load(path).get_fdata()]
save_nii_file = lambda data, path: nib.save(nib.Nifti1Image(data, np.eye(4)), path)

def read_json_file(path : str)->dict[any, any]:
    """
    Read a JSON file from the specified path and return its content.

    Args:
        path (str): The path to the JSON file.

    Returns:
        dict[any, any]: A dictionary object containing the content of the JSON file.

    """
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def write_json_file(path : str, data : dict[any, any]) -> None:
    """  
    Writes a dictionary to a JSON file.  

    This function serializes a dictionary into JSON format and writes it to   
    the specified file path with UTF-8 encoding. The JSON output is pretty-printed  
    with an indentation of 4 spaces and ensures that non-ASCII characters are preserved.  

    Args:  
        path (str): The path where the JSON file will be written.  
        data (Dict[Any, Any]): The dictionary to be serialized and written to the file.  

    Returns:  
        None  

    Raises:  
        IOError: If the file cannot be written.  
        TypeError: If the data is not serializable to JSON.  
    """  
    with open(path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)

def get_items_in_list_via_substrs(items_list : list[str], *substrs : str) -> list[str]:
    """  
    Checks if each item in the given items_list contains all specified substrings.  

    Args:  
        items_list (list[str]): List of items (strings) to be checked.  
        *substrs (str): Variable length argument list of substrings to check within each item.  

    Returns:  
        list[str]: A list of items that contain all the specified substrings.  
        
    Raises:  
        TypeError: If substrs is not provided.  

    Example:  
        >>> items = [  
        ...     "This is a test string",  
        ...     "Another example",  
        ...     "Test with multiple substrings",  
        ...     "Final string example"  
        ... ]  
        >>> get_items_in_list_via_substrs(items, "test", "string")  
        ['This is a test string', 'Test with multiple substrings']  
    """  
    results_list = [string for string in items_list if all(substr in string for substr in substrs)]
    return results_list

def merge_dicts_if_no_conflict(*dicts: dict) -> dict | None:  
    """  
    Check if there are key conflicts between multiple dictionaries.  
    If there are no key conflicts, return the merged dictionary;  
    if there are conflicts, return None.  
    
    Args:  
        *dicts: A variable number of dictionaries  

    Returns:  
        The merged dictionary or None  
    """  
    merged_dict = {}  
    for d in dicts:  
        # Check for key conflicts  
        if any(key in merged_dict for key in d):  
            return None  # There are conflicts, return None  
        # No conflicts, merge the current dictionary  
        merged_dict.update(d)  
    
    # Ensure the merged dictionary has the correct length  
    total_length = sum(len(d) for d in dicts)  
    assert len(merged_dict) == total_length, (  
        f'Error: Merged dictionary has different length than the sum of the original dictionaries.'  
    )  
    
    return merged_dict

def get_file_size(file_path : str) -> str:
    """
    """
    size_bytes = os.path.getsize(file_path)  
    units = ['B', 'KB', 'MB', 'GB', 'TB']  
    size = size_bytes  
    unit_index = 0  
    while size >= 1024 and unit_index < len(units) - 1:  
        size /= 1024.0  
        unit_index += 1
    return f'{size:.4f} {units[unit_index]}.'

''' paths '''
root_dir = '..'
NSD_dir_path = join_paths(root_dir, 'dataset', 'NSD')
fMRI_Shape_dir_path = join_paths(root_dir, 'dataset', 'fMRI_Shape')
run_saved_dir_path = join_paths(root_dir, 'run_saved')
os.makedirs(run_saved_dir_path, exist_ok=True)
train_results_dir_path = join_paths(run_saved_dir_path, 'train_results')
os.makedirs(train_results_dir_path, exist_ok=True)
test_results_dir_path = join_paths(run_saved_dir_path, 'test_results')
os.makedirs(test_results_dir_path, exist_ok=True)
NSD_saved_dir_path = join_paths(run_saved_dir_path, 'NSD_preprocessed_pairs')
os.makedirs(NSD_saved_dir_path, exist_ok=True)
fmrishape_saved_dir_path = join_paths(run_saved_dir_path, 'fMRIShape_preprocessed_pairs')
os.makedirs(fmrishape_saved_dir_path, exist_ok=True)
# sam2_ckpt_dir_path = join_paths(root_dir, 'large_files_for_BraVO', 'SAM2')
# for NSD subj  
nsd_subject_saved_dir_path = join_paths(NSD_saved_dir_path, f"subj{str(configs_dict['subj_id']).zfill(2)}")
os.makedirs(nsd_subject_saved_dir_path, exist_ok=True)
run_files_path = join_paths(nsd_subject_saved_dir_path, 'run_files.json')
# for fMRIShape subj  
fmrishape_subject_saved_dir_path = join_paths(fmrishape_saved_dir_path, f"subj{str(configs_dict['subj_id']).zfill(2)})")
os.makedirs(fmrishape_subject_saved_dir_path, exist_ok=True)


''' priori of BLIP '''
class BLIP_Prior_Tools:
    embedding_length = 768
    prefix_queries_num = 2
    img_queries_num  = 16
    txt_queries_num  = 58
    suffix_queries_num = 1
    all_embeddings_num = prefix_queries_num + img_queries_num + txt_queries_num + suffix_queries_num

    @staticmethod
    def split_hidden_states(tensor : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if tensor.size(0) == 1:
            tensor = tensor.squeeze(0)
        assert tensor.shape == (BLIP_Prior_Tools.all_embeddings_num, BLIP_Prior_Tools.embedding_length), f'{tensor.shape} != (77, 768)'
        prefix, image_embedding, caption_embedding, suffix = torch.split(tensor, 
                                                             [BLIP_Prior_Tools.prefix_queries_num, 
                                                              BLIP_Prior_Tools.img_queries_num, 
                                                              BLIP_Prior_Tools.txt_queries_num,
                                                              BLIP_Prior_Tools.suffix_queries_num
                                                             ]) 
        assert prefix.shape == (BLIP_Prior_Tools.prefix_queries_num, BLIP_Prior_Tools.embedding_length), f'prefix.shape={prefix.shape} != (2, 768)'
        assert image_embedding.shape == (BLIP_Prior_Tools.img_queries_num, BLIP_Prior_Tools.embedding_length), f'image_embedding.shape={image_embedding.shape} != (16, 768)'
        assert caption_embedding.shape  == (BLIP_Prior_Tools.txt_queries_num, BLIP_Prior_Tools.embedding_length), f'caption_embedding.shape={caption_embedding.shape} != (58, 768)'
        assert suffix.shape == (BLIP_Prior_Tools.suffix_queries_num, BLIP_Prior_Tools.embedding_length), f'suffix.shape={suffix.shape} != (1, 768)'
        return prefix, image_embedding, caption_embedding, suffix
    