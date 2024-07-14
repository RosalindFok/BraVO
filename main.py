import torch
from torch.utils.data import DataLoader

from config import configs_dict
from dataset import make_paths_dict, NSD_Dataset
from models import device, BraVO_Decoder, BraVO_Encoder#, blip_diffusion_model


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
    for index, fmri_data, image_data, image_embedding, caption_embedding, multimodal_embedding in train_dataloader:
        # Load data to device and set the dtype as float32
        tensors = [fmri_data, image_data, image_embedding, caption_embedding, multimodal_embedding]
        tensors = list(map(lambda t: t.to(device=device, dtype=torch.float32), tensors))
        fmri_data, image_data, image_embedding, caption_embedding, multimodal_embedding = tensors
        # Forward

    return model
        
def test(
    device : torch.device,
    model : torch.nn.Module,
    test_dataloader : DataLoader        
) -> None:
    model.eval()
    with torch.no_grad():
        for index, fmri_data, image_data, image_embedding, caption_embedding, multimodal_embedding in test_dataloader:
            # Load data to device and set the dtype as float32
            tensors = [fmri_data, image_data, image_embedding, caption_embedding, multimodal_embedding]
            tensors = list(map(lambda t: t.to(device=device, dtype=torch.float32), tensors))
            fmri_data, image_data, image_embedding, caption_embedding, multimodal_embedding = tensors
            



def main() -> None:
    # Hyperparameters
    batch_size = configs_dict['batch_size']
    subj_id = configs_dict['subj_id']

    # 测试 TODO blip_diffusion set
    # https://github.com/salesforce/LAVIS/tree/main/projects/blip-diffusion
    from utils import join_paths
    from PIL import Image
    from models import blip_diffusion_model, vis_preprocess, txt_preprocess
    cond_image = Image.open(join_paths('..','BraVO_saved','subj01_pairs','test','session01_run01_trial01','image.png')).convert("RGB")
    cond_images = vis_preprocess["eval"](cond_image).unsqueeze(0).cuda()
    iter_seed = 88888
    guidance_scale = 7.5
    num_inference_steps = 50
    negative_prompt = "over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate"
    cond_subjects = [txt_preprocess["eval"]('cows')]
    tgt_subjects = [txt_preprocess["eval"]('cows')]
    captions = [
        "White cows eating grass under trees and the sky",
        "Many cows in a pasture with trees eating grass.",
        "A herd of cows graze on a field of sparse grass.",
        "a herd of white cows grazing on brush among the trees",
        "A herd of mostly white cows in a field with some trees."
    ]
    for idx, caption in enumerate(captions):
        text_prompt = [txt_preprocess["eval"](caption)]
        samples = {
            "cond_images": cond_images,
            "cond_subject": cond_subjects,
            "tgt_subject": tgt_subjects,
            "prompt": text_prompt,
        }
        output = blip_diffusion_model.generate(
            samples,
            seed=iter_seed,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            neg_prompt=negative_prompt,
            height=512,
            width=512,
        )
        output[0].save(f"output_{idx}.png")
    

    # TODO 测试 https://github.com/SHI-Labs/Versatile-Diffusion

    # TODO 关注 https://arxiv.org/pdf/2311.00265

    # TODO DiT facebook DiT  https://github.com/facebookresearch/DiT

    # MMDiT
    exit(0)

    # Data
    train_paths_dict = make_paths_dict(subj_id=subj_id, mode='train')
    test_paths_dict = make_paths_dict(subj_id=subj_id, mode='test')
    train_dataloader = DataLoader(NSD_Dataset(train_paths_dict), batch_size=batch_size, shuffle=False, num_workers=6)
    test_dataloader = DataLoader(NSD_Dataset(test_paths_dict), batch_size=batch_size, shuffle=False, num_workers=6)
    
    # TODO https://github.com/salesforce/LAVIS/tree/main/projects/blip-diffusion
    # BLIP-Diffusion: Pre-trained Subject Representation for Controllable Text-to-Image Generation and Editing
    
    # Network
    model = BraVO_Encoder()

    # Loss function

    # Train
    trained_model = train(device=device, model=model, loss_fn=None, optimizer=None, train_dataloader=train_dataloader)

    # Test TODO just test blip_diffusion_model
    test(device=device, model=blip_diffusion_model, test_dataloader=test_dataloader)

if __name__ == '__main__':
    main()