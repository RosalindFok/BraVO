"""
Set paths and utility functions
"""
import os
import json
import numpy as np
import nibabel as nib

__all__ = ['join_paths', 'read_nii_file', 'save_nii_file', 'check_and_make_dirs', 'read_json_file', 'merge_dicts_if_no_conflict'
           'NSD_dir_path', 'BraVO_saved_dir_path', 'bert_base_uncased_dir_path']

''' utility functions '''
join_paths = lambda *args: os.path.join(*args)
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

''' paths '''
NSD_dir_path = join_paths('..', 'dataset', 'NSD')
BraVO_saved_dir_path = join_paths('..', 'BraVO_saved')
check_and_make_dirs(BraVO_saved_dir_path)

large_files_for_BraVO_dir_path = join_paths(os.getcwd(), '..', 'large_files_for_BraVO')
bert_base_uncased_dir_path = join_paths(large_files_for_BraVO_dir_path, 'bert-base-uncased')