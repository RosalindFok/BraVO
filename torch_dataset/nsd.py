# -*- coding: utf-8 -*-

""" 
    NSD: Natural Scenes Dataset
    Cite Info: Allen, E.J., St-Yves, G., Wu, Y. et al. A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial intelligence. Nat Neurosci 25, 116–126 (2022). https://doi.org/10.1038/s41593-021-00962-x
"""

from torch_dataset import Dataset, nsd_path, hdf5_dir

class nsd(Dataset):
    def __init__(self) -> None:
        pass
    def __getitem__(self, index) -> None:
        pass
    def __len__(self) -> int:
        return 0