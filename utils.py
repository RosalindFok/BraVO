"""
Set paths and utility functions
"""
import os
import re
import nibabel as nib


__all__ = ['join_paths', 'extract_number_from_string', 'read_nii_file', 'sort_dict_via_keys'
           'NSD_dir_path']

''' utility functions '''
join_paths = lambda *args: os.path.join(*args)
read_nii_file = lambda path: [nib.load(path).header, nib.load(path).get_fdata()]
sort_dict_via_keys = lambda d: {key : d[key] for key in sorted(d.keys())}

def extract_number_from_string(string : str) -> int | None:
    match = re.search(r'\d+', string)
    if match:
        return int(match.group())
    else:
        return None

''' paths '''
NSD_dir_path = join_paths('..', 'dataset', 'NSD')
