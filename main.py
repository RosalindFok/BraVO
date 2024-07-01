import torch
from torch.utils.data import DataLoader

from models import device
from config import configs_dict
from dataset import NSD_Dataset
from utils import read_json_file



def train(
    device : torch.device,
    model : torch.nn.Module,
    loss_fn : torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    train_dataloader : DataLoader
) -> torch.nn.Module:
    """
    Train the model using the given parameters.

    Args:
        device (torch.device): The device on which to perform training.
        model (torch.nn.Module): The neural network model to train.
        loss_fn (torch.nn.Module): The loss function used for optimization.
        optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
        train_dataloader (DataLoader): The dataloader providing training data.

    Returns:
        the updated model.
    """
    model.train()
    torch.set_grad_enabled(True)
    for fmri_data, image_data, info_path in train_dataloader:
        fmri_data  = fmri_data.to(device=device, dtype=torch.float32)
        image_data = image_data.to(device=device, dtype=torch.float32)
        info_data  = read_json_file(info_path)  # TODO 这个样子是行不通的喔
        print(info_data)

def test(
    device : torch.device,
    model : torch.nn.Module,
    test_dataloader : DataLoader        
) -> None:
    model.eval()
    with torch.no_grad():
        for fmri_data, image_data, info_path in test_dataloader:
            fmri_data = fmri_data.to(device=device, dtype=torch.float32)
            image_data = image_data.to(device=device, dtype=torch.float32)
            info_data = read_json_file(info_path)

def main() -> None:
    # Hyperparameters
    batch_size = configs_dict['batch_size']

    # Data
    train_dataloader = DataLoader(NSD_Dataset(subj_id=1, mode='train'), batch_size=batch_size, shuffle=False, num_workers=1)
    test_dataloader = DataLoader(NSD_Dataset(subj_id=1, mode='test'), batch_size=batch_size, shuffle=False, num_workers=1)
    
    # TODO BrainDiVE BrainSCUBA MindDiffuser MindEye
    # TODO Awesome CLIP and BLIP: fMRI、image、caption归约到同一个embedding空间
    # 这个太老了 看有没有新的 https://github.com/yzhuoning/Awesome-CLIP
    # TODO DiT系列生成模型 优先试试MDTv2  这两个都是ImageNet上的预训练模型啊啊啊
    # Scalable Diffusion Models with Transformers
    # https://github.com/facebookresearch/DiT
    # MDTv2    在DiT的基础上，引入了mask latent modeling，进一步提升了DiT的收敛速度和生成效果。
    # https://github.com/sail-sg/MDT
    
    # Network
    model = None

    # Loss function

    # Train
    model = train(device=device, model=model, loss_fn=None, optimizer=None, train_dataloader=train_dataloader)

    # Test

if __name__ == '__main__':
    main()