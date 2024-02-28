# -*- coding: utf-8 -*-

"""
    algnauts: The Algonauts Project 2023  - How the Human Brain Makes Sense of Natural Scenes
    Cite Info: Gifford AT, Lahner B, Saba-Sadiya S, Vilas MG, Lascelles A, Oliva A, Kay K, Roig G, Cichy RM. 2023. The Algonauts Project 2023 Challenge: How the Human Brain Makes Sense of Natural Scenes. arXiv preprint, arXiv:2301.03198. DOI: https://doi.org/10.48550/arXiv.2301.03198
    Cite Info: Allen EJ, St-Yves G, Wu Y, Breedlove JL, Prince JS, Dowdle LT, Nau M, Caron B, Pestilli F, Charest I, Hutchinson JB, Naselaris T, Kay K. 2022. A massive 7T fMRI dataset to bridge cognitive neuroscience and computational intelligence. Nature Neuroscience, 25(1):116–126. DOI: https://doi.org/10.1038/s41593-021-00962-x
"""

import os
import h5py
import time
import mmcv
import numpy as np
from tqdm import tqdm
from torch import IntTensor
from torch_dataset import Dataset, algnauts_path, hdf5_dir, config_info

subj_id_list = [os.path.join(algnauts_path, x) for x in os.listdir(algnauts_path) if os.path.isdir(os.path.join(algnauts_path, x))]
subfolder_path_list = [[os.path.join(x, y) for y in os.listdir(x)] for x in subj_id_list] # [[roi_masks, test_split, training_split], ...]
algnauts_hdf5_path = os.path.join(hdf5_dir, 'algnauts.hdf5')

class algnauts(Dataset):
    '''
    单个受试者 每个受试者训练一个模型
    '''
    def __init__(self, subject_id : int) -> None:
        super().__init__()
        # 当前的受试者id
        self.subject_id = subject_id
        # 当前受试者观看的图像数目
        self.images_num = -1
        # 当前受试者观看的图像
        self.images = None
        # 当前受试者观看的图像路径
        self.images_path = None
        # 当前受试者的左脑fMRI
        self.lh_fmri = None
        # 当前受试者的右脑fMRI
        self.rh_fmri = None
        # 当前受试者的roi文件夹路径
        self.roi_path = None

        # TODO preprocess_tutorial.ipynb 中所说的每个刺激图像的AlexNet特征
        # TODO FancyBrain 能否研究人脑观测一系列图片、两个图片之间的脑活动能反映出什么，从而连接成奇妙的视频？

        # 如果不存在algnauts数据集的hdf5文件 -> 在hdf5文件写入每个受试者的刺激图像、大脑fMRI影像、roi文件路径
        if not os.path.exists(algnauts_hdf5_path):
            start_time = time.time()
            with h5py.File(algnauts_hdf5_path, 'w') as f:
                for each_subj in subfolder_path_list:
                    subj_id = each_subj[0].split(os.sep)[-2][-1]
                    assert 0 <= int(subj_id) <= 8
                    # 每个受试者整理为一个hdf5的group
                    # 每个group中包含的dataset: images, lh_fmri, rh_fmri, roi_path
                    group = f.create_group(subj_id)
                    for path in each_subj:
                        # test_split文件夹中仅有images, 无fmri. 暂不使用
                        if 'test' in path: 
                            continue
                        # training_fmri和training_images两个文件夹
                        elif 'train' in path: 
                            imgs_dir = [os.path.join(path, d) for d in os.listdir(path) if 'images' in d][0]
                            fmri_dir = [os.path.join(path, d) for d in os.listdir(path) if 'fmri' in d][0]
                            # 该数据集中所有的图片均为png格式 且 已经裁剪至425×425, 通道为3
                            # 刺激图像创建为 images 数据集; 对图像的文件名创建为 images_path_dataset 数据集. 两个数据集相对应的位置存放的path和image相对应
                            images_dataset = group.create_dataset(name='imgs', shape=(len(os.listdir(imgs_dir)), 425, 425, 3), dtype=np.dtype('uint8'))
                            images_path_dataset = group.create_dataset(name='imgs_path', shape=(len(os.listdir(imgs_dir)),), dtype=h5py.string_dtype(encoding='utf-8'))
                            
                            for i, image_path in enumerate(tqdm(os.listdir(imgs_dir), leave=True, desc=f'sub-{subj_id}')):
                                assert '.png' in image_path
                                image_png = mmcv.imread(os.path.join(imgs_dir, image_path)) 
                                images_path_dataset[i] = image_path
                                images_dataset[i] = image_png

                            # 该数据集中所有的fMRI影像均为npy格式, 2维数组 但是数组的两个维度各异
                            for fmri_path in os.listdir(fmri_dir):
                                assert '.npy' in fmri_path
                                fmri_path = os.path.join(fmri_dir, fmri_path)
                                data = np.load(fmri_path)
                                ''' Important information:
                                    shape = (stimulus images , LH/RH vertices)
                                '''
                                shape = data.shape 
                                assert len(images_dataset) == shape[0]
                                # fMRI影像创建为 lh_fmri和rh_fmri  数据集, 分别代表左半球和右半球
                                # 左半球
                                if 'lh_training_fmri' in fmri_path:
                                    lh_fmri_dataset = group.create_dataset(name='lh_fmri', shape=shape, dtype=np.dtype('float32'))
                                    lh_fmri_dataset[:] = data
                                # 右半球
                                elif 'rh_training_fmri' in fmri_path:
                                    rh_fmri_dataset = group.create_dataset(name='rh_fmri', shape=shape, dtype=np.dtype('float32'))
                                    rh_fmri_dataset[:] = data
                        # 32个npy格式文件
                        elif 'roi' in path:
                            # 所有roi文件以npy结尾, 将其路径创建为 roi_path 数据集 
                            roi_paths_list = [os.path.join(path, x) for x in os.listdir(path)]
                            roi_path_dataset = group.create_dataset(name='roi_path', shape=(len(roi_paths_list),), dtype=h5py.string_dtype(encoding='utf-8'))
                            for i, roi_path in enumerate(roi_paths_list):
                                roi_path_dataset[i] = roi_path
            end_time = time.time()
            print(f'It took {round((end_time-start_time)/60, 2)} minutes to generate {algnauts_hdf5_path}.')
    
        # 读取algnauts数据集的hdf5文件中的内容
        print(f'Now is reading HDF5: {algnauts_hdf5_path}.')
        start_time = time.time()
        with h5py.File(algnauts_hdf5_path, 'r') as f:
            data = f[str(self.subject_id)]
            assert len(data['imgs'][:])==len(data['imgs_path'][:])==len(data['lh_fmri'][:])==len(data['rh_fmri'][:])
            self.images_num = len(data['imgs'][:])
            self.images = data['imgs'][:]
            self.images_path = data['imgs_path'][:]
            self.lh_fmri = data['lh_fmri'][:]
            self.rh_fmri = data['rh_fmri'][:]
            self.roi_path = data['roi_path'][:].tolist()*self.images_num
        end_time = time.time()
        print(f'It took {round((end_time-start_time)/60, 2)} minutes to read {algnauts_hdf5_path}.\n')
    def __getitem__(self, index): # index = range(0, __len__'s return value)
        # 返回 单张图像内容 该图像路径 该图像在左脑对应的fMRI 该图像在右脑对应的fMRI roi数据的路径


        # TODO 图像数据的排序 跟fMRI中影像的排序怎么对应得上的？？？？？？？


        return (
                    self.images[index], 
                    self.images_path[index], 
                    self.lh_fmri[index], 
                    self.rh_fmri[index], 
                    self.roi_path[index]
                )
        
        # 可能需要data['imgs'][:][index]这样的
    def __len__(self) -> int:
        assert self.images_num > 0
        return self.images_num
    