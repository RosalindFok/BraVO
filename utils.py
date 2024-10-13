"""
Set paths and utility functions
"""
import os
import json
import numpy as np
import nibabel as nib

from config import configs_dict

''' utility functions '''
join_paths = lambda *args: os.path.abspath(os.path.join(*args))
read_nii_file = lambda path: [nib.load(path).header, nib.load(path).get_fdata()]
save_nii_file = lambda data, path: nib.save(nib.Nifti1Image(data, np.eye(4)), path)
check_and_make_dirs = lambda path: os.makedirs(path, exist_ok=True)

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

def merge_dicts_if_no_conflict(dict1 : dict[any, any], dict2 : dict[any, any]) -> dict[any, any] | None:
    """
    Check if there are key conflicts between two dictionaries.
    If there are no key conflicts, return the merged dictionary;
    if there are conflicts, return None.
    
    Args:
        dict1: The first dictionary
        dict2: The second dictionary

    Returns:
        The merged dictionary or None
    """
    # Check for key conflicts
    if any(key in dict1 for key in dict2):
        return None  # There are conflicts, return None
    # No conflicts, merge the dictionaries
    merged_dict = {**dict1, **dict2}
    assert len(merged_dict) == len(dict1) + len(dict2), print(f'Error: Merged dictionary has different length than the sum of the original dictionaries.')
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
NSD_dir_path = join_paths('..', 'dataset', 'NSD')
fMRI_Shape_dir_path = join_paths('..', 'dataset', 'fMRI_Shape')
run_saved_dir_path = join_paths('..', 'run_saved')
check_and_make_dirs(run_saved_dir_path)
train_results_dir_path = join_paths(run_saved_dir_path, 'train_results')
check_and_make_dirs(train_results_dir_path)
test_results_dir_path = join_paths(run_saved_dir_path, 'test_results')
check_and_make_dirs(test_results_dir_path)
NSD_saved_dir_path = join_paths(run_saved_dir_path, 'NSD_preprocessed_pairs')
check_and_make_dirs(NSD_saved_dir_path)
fmrishape_saved_dir_path = join_paths(run_saved_dir_path, 'fMRIShape_preprocessed_pairs')
check_and_make_dirs(fmrishape_saved_dir_path)
sam2_ckpt_dir_path = join_paths('..', 'large_files_for_BraVO', 'SAM2')
# for NSD subj  
nsd_subject_saved_dir_path = join_paths(NSD_saved_dir_path, f"subj{str(configs_dict['subj_id']).zfill(2)}")
check_and_make_dirs(nsd_subject_saved_dir_path)
run_files_path = join_paths(nsd_subject_saved_dir_path, 'run_files.json')
# for fMRIShape subj  
fmrishape_subject_saved_dir_path = join_paths(fmrishape_saved_dir_path, f"subj{str(configs_dict['subj_id']).zfill(2)})")
check_and_make_dirs(fmrishape_subject_saved_dir_path)


