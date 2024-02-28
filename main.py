# -*- coding: utf-8 -*-

"""
    Info about main.py
""" 

from torch.utils.data import DataLoader
from torch_dataset import algnauts, config_info, config_path

def main() -> None:
    # 加载当前实验所使用的数据集
    dataset_name = config_info['dataset_name']
    print(f'{"-"*8}Now is working on dataset {dataset_name}.{"-"*8}')
    # 选择当前所使用数据集的被试者群体编号
    selected_subjects_list = config_info['subjects'][dataset_name] 

    # 每个受试者需要训练一个单独的模型
    for subject_id in selected_subjects_list:
        dataset = algnauts.algnauts(subject_id=subject_id)
        dataloader = DataLoader(dataset=dataset)
    
        for imgs, imgs_path, lh_fmri, rh_fmri, roi_path in dataloader:
            pass
if __name__ == '__main__':
    main()