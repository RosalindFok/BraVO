import torch
from utils import join_paths
from dataset import NSD_HDF5_Dataset
from torch.utils.data import DataLoader

dir_path = join_paths('..', 'BraVO_saved', 'NSD_preprocessed_pairs', 'subj01_pairs', 'hdf5-V1v_V1d')
image_train_hdf5_path = join_paths(dir_path,  'train_data.hdf5')
image_test_hdf5_path = join_paths(dir_path, 'test_data.hdf5')

train_dataset = NSD_HDF5_Dataset(hdf5_path=image_train_hdf5_path)
test_dataset = NSD_HDF5_Dataset(hdf5_path=image_test_hdf5_path)


train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

for batchs in train_dataloader:
    # print(batchs['masked_embedding'].shape)
    # print(batchs['hidden_states_image'].shape)
    image_data = batchs.hidden_states_image
    # 打印image_data最大最小值
    print("Maximum value:", torch.max(image_data).item())
    print("Minimum value:", torch.min(image_data).item())

    # print("Number of elements greater than 2:", torch.sum(image_data > 2).item())
    # print("Number of elements less than -2:", torch.sum(image_data < -2).item())
    # print("Number of elements between 0 and 1:", torch.sum((image_data >= 0) & (image_data <= 1)).item())
    # print("Number of elements between -1 and 0:", torch.sum((image_data >= -1) & (image_data < 0)).item())
    # print("Number of elements between 1 and 2:", torch.sum((image_data >= 1) & (image_data <= 2)).item())
    # print("Number of elements between -1 and -2:", torch.sum((image_data >= -2) & (image_data <= -1)).item())

