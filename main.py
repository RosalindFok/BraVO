import torch
from torch.utils.data import DataLoader

from config import configs_dict
from dataset import NSD_Dataset

def setup_device() -> torch.device:
    """
    Set up and return the available torch device.

    Returns:
        torch.device: A torch.device object representing the device,
        choosing GPU or CPU based on the availability of CUDA.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device = {device if not torch.cuda.is_available() else torch.cuda.get_device_name(torch.cuda.current_device())}')
    torch.cuda.init() if torch.cuda.is_available() else None
    return device

def main() -> None:
    # Device
    device = setup_device()

    # Hyperparameters
    batch_size = configs_dict['batch_size']

    # Data
    train_loader = DataLoader(NSD_Dataset(subj_id=1, mode='train'), batch_size=batch_size, shuffle=False, num_workers=1)
    test_loader = DataLoader(NSD_Dataset(subj_id=1, mode='test'), batch_size=batch_size, shuffle=False, num_workers=1)
    
    # TODO BrainDiVE BrainSCUBA MindDiffuser MindEye2
    # TODO Awesome CLIP and BLIP: fMRI、image、caption归约到同一个embedding空间
    # TODO DiT系列生成模型 优先试试MDTv2  这两个都是ImageNet上的预训练模型啊啊啊
    # Scalable Diffusion Models with Transformers
    # https://github.com/facebookresearch/DiT
    # MDTv2    在DiT的基础上，引入了mask latent modeling，进一步提升了DiT的收敛速度和生成效果。
    # https://github.com/sail-sg/MDT
    
    # Network

if __name__ == '__main__':
    main()