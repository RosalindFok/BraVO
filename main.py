# -*- coding: utf-8 -*-

"""
    Info about main.py
""" 

import sys
import torch
from torch.utils.data import DataLoader
from torch_dataset import algnauts, nod, nsd, config_info

def main() -> None:
    # 加载当前实验所使用的数据集
    dataset_name = config_info['dataset_name']
    print(f'{"-"*8}Now is working on dataset {dataset_name}.{"-"*8}')
    # 选择当前所使用数据集的被试者群体编号
    selected_subjects_list = config_info['subjects'][dataset_name] 

    # 每个受试者需要训练一个单独的模型
    for subject_id in selected_subjects_list:
        
        ### algnauts 2023 ###
        if dataset_name.lower() == 'algnauts':
            # 加载DataLoader
            dataset = algnauts.algnauts(subject_id=subject_id)
            batch_size = len(dataset) if config_info['train']['batch_size'] == None else config_info['train']['batch_size']
            dataloader = DataLoader(dataset=dataset, batch_size=batch_size)

            # 该受试者观看的图像数目
            images_num = len(dataset) # 应当注意 len(dataloader) = 总数目 // batch_size
            print(f'Subject - {subject_id} watched {images_num} images.')

            for imgs, imgs_path, lh_fmri, rh_fmri, roi_path in dataloader:
                ### DataSet返回值类型与形状 检验点
                assert isinstance(imgs, torch.Tensor) and imgs.shape == torch.Size([batch_size, 425, 425, 3])
                assert isinstance(imgs_path, tuple) and all(isinstance(x, bytes) for x in imgs_path) and len(imgs_path) == batch_size
                assert isinstance(lh_fmri, torch.Tensor) and lh_fmri.shape[0] == images_num
                assert isinstance(rh_fmri, torch.Tensor) and rh_fmri.shape[0] == images_num
                assert isinstance(roi_path, tuple) and len(roi_path) == batch_size
                ### DataSet返回值类型与形状 检验点
        
        ### NOD ###
        elif dataset_name.lower() == 'nod':
            # 加载DataLoader
            dataset = nod.nod(subject_id=subject_id)
            # batch_size = len(dataset) if config_info['train']['batch_size'] == None else config_info['train']['batch_size']
            # dataloader = DataLoader(dataset=dataset, batch_size=batch_size)

        
        ### NSD ###
        elif dataset_name.lower() == 'nsd':
            pass

        ### 暂未处理的数据集 ###
        else:
            sys.stderr.write(f'The dataset {dataset_name} has not been included!')
        
if __name__ == '__main__':
    main()