# -*- coding: utf-8 -*-

"""
    Info about main.py
""" 

import torch_dataset.algnauts as data
from torch.utils.data import DataLoader

def main() -> None:
    dataset = data.algnauts()
    dataloader = DataLoader(dataset=dataset)
    for x in dataloader:
        print(x)

if __name__ == '__main__':
    main()