import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from config import configs_dict
from utils import read_json_file
from dataset import make_paths_dict, NSD_Dataset
from models import device, BraVO_Decoder, BraVO_Encoder, load_blip_models


def train(
    device : torch.device,
    model : torch.nn.Module,
    loss_fn : torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    train_dataloader : DataLoader
) -> torch.nn.Module:
    """
    
    """
    model.train()
    torch.set_grad_enabled(True)
    for index, masked_data, image_data, embedding in tqdm(train_dataloader, desc='Training', leave=True):
        # Load data to device and set the dtype as float32
        tensors = [masked_data, image_data, embedding]
        tensors = list(map(lambda t: t.to(device=device, dtype=torch.float32), tensors))
        masked_data, image_data, embedding = tensors
        # Forward

    return model
        
def test(
    device : torch.device,
    model : torch.nn.Module,
    test_dataloader : DataLoader        
) -> None:
    """
    
    """
    model.eval()
    with torch.no_grad():
        for index, masked_data, image_data, embedding in tqdm(test_dataloader, desc='Testing', leave=True):
            # Load data to device and set the dtype as float32
            tensors = [masked_data, image_data, embedding]
            tensors = list(map(lambda t: t.to(device=device, dtype=torch.float32), tensors))
            masked_data, image_data, embedding = tensors
            # Forward
            

def mask(derived_type : str, roi_name : str, rois_path_dict : dict[str, dict[str, list[str]]]) -> tuple[list[str], dict[str, str]]:
    derived_type = derived_type.lower()
    roi_name = roi_name.lower()
    if not derived_type in rois_path_dict.keys():
        raise ValueError(f'derived_type should be one of {rois_path_dict.keys()}, but got {derived_type}')
    if not roi_name in rois_path_dict[derived_type].keys():
        raise ValueError(f'roi_name should be one of {rois_path_dict[derived_type].keys()}, but got {roi_name}')
    # 4 = roi_name.nii.gz, lh.name.nii.gz, rh.name.nii.gz, label_tags.json
    files_path_list = rois_path_dict[derived_type][roi_name]
    assert len(files_path_list) == 4, print(f'{files_path_list}')
    json_path = [f for f in files_path_list if f.endswith('.json')][0]
    files_path_list.remove(json_path)
    label_tags = read_json_file(json_path)
    return files_path_list, label_tags


def main() -> None:
    # Hyperparameters
    batch_size = configs_dict['batch_size']
    subj_id = configs_dict['subj_id']
    derived_type = configs_dict['derived_type']
    roi_name = configs_dict['roi_name']
    threshold = configs_dict['threshold']

    # from utils import join_paths
    # from PIL import Image
    # blip_diffusion_model, vis_preprocess, txt_preprocess = load_blip_models(mode = 'diffusion')
    # cond_image = Image.open(join_paths('..','BraVO_saved','subj01_pairs','test','session01_run03_trial42','image.png')).convert("RGB")
    # cond_images = vis_preprocess["eval"](cond_image).unsqueeze(0).to(device)
    # iter_seed = 88888
    # guidance_scale = 7.5
    # num_inference_steps = 500 # TODO 可以调整哒
    # negative_prompt = "over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate"
    # cond_subjects = [txt_preprocess["eval"]('person')]
    # tgt_subjects = [txt_preprocess["eval"]('person')]
    # captions = [
    #     "A person holding a remote control in their hand.",
    #     "a person is holding a remote with a blanket over them.",
    #     "a person about to hit a button on a remote control",
    #     "A person holding a remote as it rests on a blanket.",
    #     "Someone is holding a grey remote control near a blanket."
    # ]
    # for idx, caption in enumerate(captions):
    #     text_prompt = [txt_preprocess["eval"](caption)]
    #     samples = {
    #         "cond_images": cond_images,
    #         "cond_subject": cond_subjects,
    #         "tgt_subject": tgt_subjects,
    #         "prompt": text_prompt,
    #     }
    #     output = blip_diffusion_model.generate_embedding(
    #         samples=samples,
    #         guidance_scale=guidance_scale,
    #         neg_prompt=negative_prompt,
    #     )
    #     print(output.shape)
    #     output = blip_diffusion_model.generate_image_via_embedding(
    #         text_embeddings=output,
    #         seed=iter_seed,
    #         guidance_scale=guidance_scale,
    #         height=512,
    #         width=512,
    #         num_inference_steps=num_inference_steps,
    #     )
    #     output[0].save(f"output_{idx}.png")

    

    # # TODO DiT facebook DiT  https://github.com/facebookresearch/DiT    有没有two guided的DiT 或者学习人家blip-diffusion把图像+文本来生成一个text embedding

    # Data
    train_trial_path_dict, test_trial_path_dict, rois_path_dict = make_paths_dict(subj_id=subj_id)
    mask_path_list, label_tags = mask(derived_type=derived_type, roi_name=roi_name, rois_path_dict=rois_path_dict)
    rois_name_list = [value for key, value in label_tags.items() if int(key) > threshold]
    train_dataloader = DataLoader(NSD_Dataset(train_trial_path_dict, mask_path_list, threshold=threshold), 
                                  batch_size=batch_size, shuffle=False, num_workers=1)
    test_dataloader = DataLoader(NSD_Dataset(test_trial_path_dict, mask_path_list, threshold=threshold), 
                                 batch_size=batch_size, shuffle=False, num_workers=1)
    
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