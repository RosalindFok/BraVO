import os
import time
import copy
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from config import configs_dict
from models import device, BraVO_Encoder
from dataset import make_paths_dict, NSD_Dataset, masking_fmri_to_array
from utils import join_paths, read_json_file, write_json_file, read_nii_file, BraVO_saved_dir_path, check_and_make_dirs
from analyze import analyze_volume, analyze_volume_MTL, analyze_volume_thalamus, analyze_surface_floc

def train(
    device : torch.device,
    model : torch.nn.Module,
    loss_fn : torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    train_dataloader : DataLoader
) -> tuple[torch.nn.Module, float]:
    """
    Trains the neural network model for one epoch.  

    Args:  
        device (torch.device): The device (CPU or GPU) to run the model on.  
        model (torch.nn.Module): The neural network model to be trained.  
        loss_fn (torch.nn.Module): The loss function to compute the error.  
        optimizer (torch.optim.Optimizer): The optimizer to adjust the model parameters.  
        train_dataloader (DataLoader): DataLoader object to fetch training data in batches.  

    Returns:  
        tuple: A tuple containing the trained model and the average training loss for the epoch.  
    """
    model.train()
    torch.set_grad_enabled(True)
    train_loss = []
    for index, fmri_data, masked_data, image_data, embedding in tqdm(train_dataloader, desc='Training', leave=True):
        # Load data to device and set the dtype as float32
        tensors = [fmri_data, masked_data, image_data, embedding]
        tensors = list(map(lambda t: t.to(device=device, dtype=torch.float32), tensors))
        fmri_data, masked_data, image_data, embedding = tensors
        # Forward
        pred_brain = model(embedding)
        loss = loss_fn(pred_brain, fmri_data)
        train_loss.append(loss.item())
        # 3 steps of back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model, sum(train_loss)/len(train_loss)


def test(
    device : torch.device,
    model : torch.nn.Module,
    metrics_fn : dict[str, torch.nn.Module],
    test_dataloader : DataLoader,
    saved_pred_brain_dir_path : str        
) -> None:
    """
    Evaluates the model on the test dataset and computes specified metrics.  

    Args:  
        device (torch.device): The device (CPU or GPU) to run the model on.  
        model (torch.nn.Module): The neural network model to be tested.  
        metrics_fn (dict[str, torch.nn.Module]): A dictionary containing the metric functions to compute.  
        test_dataloader (DataLoader): DataLoader object to load the test dataset.  
        saved_pred_brain_dir_path (str): Directory path to save prediction results.  

    Returns:  
        None  
    """
    model.eval()
    metrics = {key : [] for key in metrics_fn.keys()}
    with torch.no_grad():
        for index, fmri_data, masked_data, image_data, embedding in tqdm(test_dataloader, desc='Testing', leave=True):
            # Load data to device and set the dtype as float32
            tensors = [fmri_data, masked_data, image_data, embedding]
            tensors = list(map(lambda t: t.to(device=device, dtype=torch.float32), tensors))
            fmri_data, masked_data, image_data, embedding = tensors
            # Forward
            pred_brain = model(embedding)
            for metric_name in metrics_fn:
                value = metrics_fn[metric_name](pred_brain, fmri_data)
                metrics[metric_name].append(value.item())
            # save the pred_brain
            index = index.cpu().numpy()
            fmri_data = fmri_data.cpu().numpy()
            pred_brain = pred_brain.cpu().numpy()
            for idx, fmri, pred in zip(index, fmri_data, pred_brain):
                np.save(join_paths(saved_pred_brain_dir_path, f'{str(idx)}_gt.npy'), fmri)
                np.save(join_paths(saved_pred_brain_dir_path, f'{str(idx)}_pd.npy'), pred)

    for metric_name in metrics:
        value = sum(metrics[metric_name])/len(metrics[metric_name])
        print(f'{metric_name}Loss: {value}')


def fetch_roi_files_and_labels(derived_type : str, roi_name : str, 
                               rois_path_dict : dict[str, dict[str, list[str]]]
                        ) -> tuple[list[str], dict[int, str]]:
    """  
    Fetches the file paths and labels for a given region of interest (ROI).  

    Args:  
        derived_type: A string representing the derived type. Should match one of the keys in rois_path_dict.  
        roi_name: A string representing the ROI name. Should match one of the keys in the nested dictionary for the given derived_type in rois_path_dict.  
        rois_path_dict: A dictionary where keys are derived types and values are another dictionary,  
                        where the keys are ROI names and values are lists of file paths related to the ROI.  

    Returns:  
        A tuple containing:  
            - list of strings: Paths of ROI files.  
            - dict of strings: Key-value pairs from the JSON file corresponding to label tags.  

    Raises:  
        ValueError: If derived_type is not a key in rois_path_dict.  
        ValueError: If roi_name is not a key in the nested dictionary for the given derived_type.  
    """  
    # derived_type = derived_type.lower()
    # roi_name = roi_name.lower()
    if not derived_type in rois_path_dict.keys():
        raise ValueError(f'derived_type should be one of {rois_path_dict.keys()}, but got {derived_type}')
    if not roi_name in rois_path_dict[derived_type].keys():
        raise ValueError(f'roi_name should be one of {rois_path_dict[derived_type].keys()}, but got {roi_name}')
    # 4 = roi_name.nii.gz, lh.name.nii.gz, rh.name.nii.gz, label_tags.json
    rois_path_dict_copy = copy.deepcopy(rois_path_dict)
    files_path_list = rois_path_dict_copy[derived_type][roi_name]
    assert len(files_path_list) == 4, print(f'{files_path_list}')
    json_path = [f for f in files_path_list if f.endswith('.json')][0]
    files_path_list.remove(json_path)
    label_tags = read_json_file(json_path)
    label_tags = {int(key) : value for key, value in label_tags.items()}
    return files_path_list, label_tags


def optimal_stimulus_rois(rois_path_dict : dict[str, dict[str, list[str]]],
                          test_trial_path_dict : dict[str, dict[str, str]],
                          saved_pred_brain_dir_path : str
                          ) -> dict[int, dict[str, any]]:
    """  
    Processes fMRI data to evaluate the quality of predictions using different regions of interest (ROIs).  
    
    Args:  
        rois_path_dict (dict[str, dict[str, list[str]]]): Dictionary mapping ROI types to file paths.  
        test_trial_path_dict (dict[str, dict[str, str]]): Dictionary mapping test trials to file paths.  
        saved_pred_brain_dir_path (str): Directory path to saved predicted brain data.  
        
    Returns:  
        dict[int, dict[str, any]]: Dictionary of results containing metrics and categories for each test trial.  
    """  
    results = {} 
    for idx in tqdm(range(len(os.listdir(saved_pred_brain_dir_path))//2), desc=f'Masking', leave=True):
        results[idx] = {}
        fmri_path, pred_path = join_paths(saved_pred_brain_dir_path, f'{str(idx)}_gt.npy'), join_paths(saved_pred_brain_dir_path, f'{str(idx)}_pd.npy')
        assert os.path.exists(fmri_path), print(f'{fmri_data} does not exist.')
        assert os.path.exists(pred_path), print(f'{pred_brain} does not exist.')
        fmri_data = np.load(fmri_path)
        pred_brain = np.load(pred_path)
        assert fmri_data.shape == pred_brain.shape, print(f'{fmri_data.shape} and {pred_brain.shape} should be the same.')
        # mask
        rois_type = {key : [] for key in rois_path_dict.keys()}
        for derived_type in rois_path_dict:
            rois_type[derived_type].extend(list(rois_path_dict[derived_type].keys()))
        for derived_type in rois_type:
            results[idx][derived_type] = {}
            for roi_name in rois_type[derived_type]:
                results[idx][derived_type][roi_name] = {}
                mask_path_list, label_tags = fetch_roi_files_and_labels(derived_type=derived_type, roi_name=roi_name, rois_path_dict=rois_path_dict)
                mask_header, mask_data = read_nii_file([x for x in mask_path_list if '.nii.gz' in x and not 'lh.' in x and not 'rh.' in x][0])
                max_key = max([key for key in label_tags.keys()])
                for threshold  in range(max_key):
                    region_name = 'whole' if threshold == 0 else label_tags[threshold]
                    masked_fmri = masking_fmri_to_array(fmri_data=fmri_data, mask_data=mask_data, threshold=threshold)
                    masked_pred = masking_fmri_to_array(fmri_data=pred_brain, mask_data=mask_data, threshold=threshold)
                    if masked_fmri is None and masked_pred is None: # no such a region in the mask
                        continue
                    assert masked_fmri.shape == masked_pred.shape, print(f'{masked_fmri.shape} and {masked_pred.shape} should be the same.')
                    # MSE
                    mse = np.mean((masked_fmri - masked_pred)**2)
                    # MAE
                    mae = np.mean(np.abs(masked_fmri - masked_pred))
                    # save to results dict
                    results[idx][derived_type][roi_name][region_name] = {'MSE':float(mse), 'MAE':float(mae)}
        # categories
        strings = read_json_file(test_trial_path_dict[idx]['strings'])
        selected_category = strings['selected_category']
        for categories in strings['instances_category']:
            if categories['name'] == selected_category:
                supercategory = categories['supercategory']
                break
        results[idx][supercategory] = selected_category
    
    return results
    

def analyze_masked_result(saved_masking_result_json_path : str) -> None:
    masking_result = read_json_file(saved_masking_result_json_path)
    ## Surface

    ## Volume
    volume_derived_result = analyze_volume(masking_result=masking_result)
    analyze_volume_MTL(masking_result=masking_result)
    analyze_volume_thalamus(masking_result=masking_result)
    analyze_surface_floc(masking_result=masking_result)

def main() -> None:
    ## Train or Test
    parser = argparse.ArgumentParser(description='Select from train or test.')
    parser.add_argument('--task', type=str)
    args = parser.parse_args()
    task = args.task.lower()

    ## Hyperparameters
    # subj id
    subj_id = configs_dict['subj_id']
    # train brain encoder
    batch_size = configs_dict['train_encoder']['batch_size']
    learning_rate = configs_dict['train_encoder']['learning_rate']
    epochs = configs_dict['train_encoder']['epochs']
    # roi
    derived_type = 'surface'
    roi_name = 'corticalsulc'
    threshold = 0

    # from utils import join_paths
    # from PIL import Image
    # from models import load_blip_models
    # blip_diffusion_model, vis_preprocess, txt_preprocess = load_blip_models(mode = 'diffusion')
    # cond_image = Image.open(join_paths('..','BraVO_saved','subj01_pairs','test','session01_run01_trial01','image.png')).convert("RGB")
    # cond_images = vis_preprocess["eval"](cond_image).unsqueeze(0).to(device)
    # iter_seed = 88888
    # guidance_scale = 7.5
    # num_inference_steps = 500 # TODO 可以调整哒
    # negative_prompt = "over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate"
    # cond_subjects = [txt_preprocess["eval"]('person')]
    # tgt_subjects = [txt_preprocess["eval"]('person')]
    # captions = [
    #     "White cows eating grass under trees and the sky",
    #     "Many cows in a pasture with trees eating grass.",
    #     "A herd of cows graze on a field of sparse grass.",
    #     "a herd of white cows grazing on brush among the trees",
    #     "A herd of mostly white cows in a field with some trees."

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
    #     output = blip_diffusion_model.generate_image_via_embedding(
    #         text_embeddings=output,
    #         seed=iter_seed,
    #         guidance_scale=guidance_scale,
    #         height=512,
    #         width=512,
    #         num_inference_steps=num_inference_steps,
    #     )
    #     output[0].save(f"output_{idx}.png")
    # exit(0)
    # # TODO DiT facebook DiT  https://github.com/facebookresearch/DiT    有没有two guided的DiT 或者学习人家blip-diffusion把图像+文本来生成一个text embedding

    ## Data
    train_trial_path_dict, test_trial_path_dict, rois_path_dict = make_paths_dict(subj_id=subj_id)
    mask_path_list, label_tags = fetch_roi_files_and_labels(derived_type=derived_type, roi_name=roi_name, rois_path_dict=rois_path_dict)
    train_dataloader = DataLoader(NSD_Dataset(train_trial_path_dict, mask_path_list, threshold=threshold), 
                                batch_size=batch_size, shuffle=False, num_workers=1)
    test_dataloader = DataLoader(NSD_Dataset(test_trial_path_dict, mask_path_list, threshold=threshold), 
                                batch_size=batch_size, shuffle=False, num_workers=1)

    ## Path to save
    # the path of saving the trained model
    saved_model_path = join_paths(BraVO_saved_dir_path, f'subj{str(subj_id).zfill(2)}_model_ep-{epochs}_lr-{learning_rate}_bs-{batch_size}.pth')
    # path to save the prediected fMRI(whole brain)
    saved_pred_brain_dir_path = join_paths(BraVO_saved_dir_path, f'subj{str(subj_id).zfill(2)}_pred_brain_fMRI')
    check_and_make_dirs(saved_pred_brain_dir_path)# Train
    # path to save the result of masking
    saved_masking_result_json_path = join_paths(BraVO_saved_dir_path, f'subj{str(subj_id).zfill(2)}_masking_result.json')

    ## Task
    # If th task if train or test, load the network, loss function and optimizer.
    if task in ['train', 'test']:
        # Network for Brain Encoder: input = embedding, output = predicted brain fMRI
        light_loader = next(iter(DataLoader(NSD_Dataset(train_trial_path_dict, mask_path_list, threshold=threshold), batch_size=1, shuffle=False, num_workers=1)))
        input_shape  = light_loader[-1].shape[-3:] # the shape of embedding
        output_shape = light_loader[1].shape[-3:] # the shape of fmri_data
        bravo_encoder_model = BraVO_Encoder(input_shape=input_shape, output_shape=output_shape)
        print(bravo_encoder_model)
        trainable_parameters = sum(p.numel() for p in bravo_encoder_model.parameters() if p.requires_grad)
        bravo_encoder_model = bravo_encoder_model.to(device=device)
        print(f'The number of trainable parametes is {trainable_parameters}.')

        # Loss function
        loss_of_brain_encoder = torch.nn.MSELoss()
        # Optimizer
        optimizer_of_brain_encoder = torch.optim.Adam(bravo_encoder_model.parameters(), lr=learning_rate) 
    elif task in ['mask', 'analyze']:
        # No need to use GPU
        pass
    
    # Train
    if task == 'train':
        print(f'Training Brain Encoder for {epochs} epochs.')
        for epoch in range(epochs):
            start_time = time.time()
            trained_model, train_loss = train(device=device, model=bravo_encoder_model, 
                              loss_fn=loss_of_brain_encoder, 
                              optimizer=optimizer_of_brain_encoder, 
                              train_dataloader=train_dataloader)
            end_time = time.time()
            print(f'Epoch {epoch+1}/{epochs}, {loss_of_brain_encoder.__class__.__name__}: {train_loss:.4f}, Time: {(end_time-start_time)/60:.2f} minutes.')
        # save the trained model
        torch.save(bravo_encoder_model.state_dict(), saved_model_path)
    # Test
    elif task == 'test':
        print(f'Testing Brain Encoder.')
        # load the trained model
        bravo_encoder_model.load_state_dict(torch.load(saved_model_path))
        metrics_fn = {
            'MSE' : torch.nn.MSELoss(),
            'MAE' : torch.nn.L1Loss(),
        }
        test(device=device, model=bravo_encoder_model, metrics_fn=metrics_fn, test_dataloader=test_dataloader, saved_pred_brain_dir_path=saved_pred_brain_dir_path)
    # Mask
    elif task == 'mask':
        print(f'Masking the result of Brain Encoder.')
        results = optimal_stimulus_rois(rois_path_dict=rois_path_dict, test_trial_path_dict=test_trial_path_dict, saved_pred_brain_dir_path=saved_pred_brain_dir_path)
        # save the results to a json file
        start_time = time.time()
        write_json_file(path=saved_masking_result_json_path, data=results)
        end_time = time.time()
        print(f'It took {end_time-start_time:.2f} seconds to write the optimal_stimulus_rois results.')
    # TODO 分析mask的结果？
    elif task == 'analyze':
        analyze_masked_result(saved_masking_result_json_path=saved_masking_result_json_path)
    else:
        raise ValueError(f'Task should be either train or test, but got {task}.')
    
if __name__ == '__main__':
    main()