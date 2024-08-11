import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from losses import VAE_loss
from models import device, get_GPU_memory_usage, BraVO_Decoder
from config import configs_dict, blip_diffusion_coco_embedding_priori
from dataset import make_paths_dict, fetch_roi_files_and_labels, NSD_Dataset
from utils import join_paths, read_json_file, BraVO_saved_dir_path, check_and_make_dirs
from analyze import analyze_blip_diffusion_embeddings_space, analyze_test_results



def transform_output_into_priori(pred_embedding : torch.Tensor) -> torch.Tensor:
    """
    """
    # priori
    popt = blip_diffusion_coco_embedding_priori['popt']
    amplitude, mean, std = popt['amplitude'], popt['mean'], popt['standard_deviation']
    # transform
    mean_x = torch.mean(pred_embedding, dim=(1, 2), keepdim=True) 
    std_x = torch.std(pred_embedding, dim=(1, 2), keepdim=True)
    transformed_pred_embedding = ((pred_embedding - mean_x) / std_x) * std + mean
    transformed_pred_embedding = transformed_pred_embedding * amplitude
    return transformed_pred_embedding

def train(
    device : torch.device,
    model : torch.nn.Module,
    loss_fn : torch.nn.modules.loss._Loss,
    optimizer : torch.optim.Optimizer,
    train_dataloader : DataLoader
) -> tuple[torch.nn.Module, float, float, float]:
    """
    """
    model.train()
    torch.set_grad_enabled(True)
    train_loss = []
    mem_reserved_list = []
    for index, masked_data, image_data, canny_data, multimodal_embedding in tqdm(train_dataloader, desc='Training', leave=True):
        # Load data to device and set the dtype as float32
        tensors = [masked_data, image_data, canny_data, multimodal_embedding]
        tensors = list(map(lambda t: t.to(device=device, dtype=torch.float32), tensors))
        masked_data, image_data, canny_data, multimodal_embedding = tensors
        # Forward
        pred_embedding, mean, log_var = model(masked_data)
        transformed_pred_embedding = transform_output_into_priori(pred_embedding=pred_embedding)
        loss = loss_fn(input=transformed_pred_embedding, target=multimodal_embedding, mean=mean, log_var=log_var)
        assert not torch.isnan(loss), 'loss is nan, stop training!'
        train_loss.append(loss.item())
        # 3 steps of back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Monitor GPU memory usage
        total_memory, mem_reserved = get_GPU_memory_usage()
        mem_reserved_list.append(mem_reserved)
    return model, sum(train_loss)/len(train_loss), total_memory, max(mem_reserved_list)

def test(
    device : torch.device,
    model : torch.nn.Module,
    metrics_fn : dict[str, torch.nn.Module],
    test_dataloader : DataLoader,
    saved_test_results_dir_path : str        
) -> None:
    """
    """
    model.eval()
    metrics = {key : [] for key in metrics_fn.keys()}
    with torch.no_grad():
        for index, masked_data, image_data, canny_data, multimodal_embedding in tqdm(test_dataloader, desc='Testing', leave=True):
            # Load data to device and set the dtype as float32
            tensors = [masked_data, image_data, canny_data, multimodal_embedding]
            tensors = list(map(lambda t: t.to(device=device, dtype=torch.float32), tensors))
            masked_data, image_data, canny_data, multimodal_embedding = tensors
            # Forward
            pred_embedding, mean, log_var  = model(masked_data)
            transformed_pred_embedding = transform_output_into_priori(pred_embedding=pred_embedding)
            for metric_name in metrics_fn:
                value = metrics_fn[metric_name](transformed_pred_embedding, multimodal_embedding)
                metrics[metric_name].append(value.item())
            # save the transformed_pred_embedding
            index = index.cpu().numpy()
            multimodal_embedding = multimodal_embedding.cpu().numpy()
            transformed_pred_embedding = transformed_pred_embedding.cpu().numpy()
            for idx, fmri, pred in zip(index, multimodal_embedding, transformed_pred_embedding):
                np.save(join_paths(saved_test_results_dir_path, f'{str(idx)}_true.npy'), fmri)
                np.save(join_paths(saved_test_results_dir_path, f'{str(idx)}_pred.npy'), pred)

    for metric_name in metrics:
        value = sum(metrics[metric_name])/len(metrics[metric_name])
        print(f'{metric_name}Loss: {value}')
   

def main() -> None:
    ## Train or Test
    parser = argparse.ArgumentParser(description='Select from train or test.')
    parser.add_argument('--task', type=str)
    args = parser.parse_args()
    task = args.task.lower()

    ## Hyperparameters
    # subj id
    subj_id = configs_dict['subj_id']
    # dataset name
    dataset_name = configs_dict['dataset_name']
    # train brain decoder
    batch_size = configs_dict['train_decoder']['batch_size']
    learning_rate = configs_dict['train_decoder']['learning_rate']
    epochs = configs_dict['train_decoder']['epochs']
    # roi
    derived_type = configs_dict['NSD_ROIs']['derived_type']
    roi_name = configs_dict['NSD_ROIs']['roi_name']
    thresholds = configs_dict['NSD_ROIs']['thresholds']
    
    ## Data
    train_trial_path_dict, test_trial_path_dict, rois_path_dict, uncond_embedding = make_paths_dict(subj_id=subj_id, dataset_name=dataset_name)
    mask_path_list, labels_string = fetch_roi_files_and_labels(derived_type=derived_type, roi_name=roi_name, thresholds=thresholds, rois_path_dict=rois_path_dict)
    train_dataloader = DataLoader(dataset=NSD_Dataset(train_trial_path_dict, mask_path_list, thresholds), 
                                batch_size=batch_size, shuffle=False, num_workers=1)
    test_dataloader = DataLoader(dataset=NSD_Dataset(test_trial_path_dict, mask_path_list, thresholds), 
                                batch_size=batch_size, shuffle=False, num_workers=1)

    ## Path to save
    # the path of saving the trained model
    ckpt_dir_path = join_paths(BraVO_saved_dir_path, f'subj{str(subj_id).zfill(2)}_checkpoints')
    check_and_make_dirs(ckpt_dir_path)
    saved_model_path = join_paths(BraVO_saved_dir_path, f'subj{str(subj_id).zfill(2)}_model_ep-{epochs}_lr-{learning_rate}_bs-{batch_size}.pth')
    # path to save the prediected fMRI(whole brain)
    saved_test_results_dir_path = join_paths(BraVO_saved_dir_path, f'subj{str(subj_id).zfill(2)}_test_results')
    check_and_make_dirs(saved_test_results_dir_path)# Train

    ## Algorithm
    if task in ['train', 'test']: # to save memory usage
        # Network for Brain Decoder: input = embedding, output = predicted brain fMRI
        light_loader = next(iter(test_dataloader))
        input_shape  = light_loader[1].shape[1:]  # The shape of masked_data
        output_shape = light_loader[-1].shape[1:] # The shape of multimodal_embedding
        bravo_decoder_model = BraVO_Decoder(input_shape=input_shape, output_shape=output_shape)
        print(bravo_decoder_model)
        trainable_parameters = sum(p.numel() for p in bravo_decoder_model.parameters() if p.requires_grad)
        bravo_decoder_model = bravo_decoder_model.to(device=device)
        print(f'The number of trainable parametes is {trainable_parameters}.')
        # Loss function
        loss_of_brain_decoder = VAE_loss(w_mse=1, w_kld = 1)
        # Optimizer
        optimizer_of_brain_decoder = torch.optim.Adam(bravo_decoder_model.parameters(), lr=learning_rate) 
    elif task in ['analyze']:
        pass

    # Train
    if task == 'train':
        print(f'Training Brain Decoder for {epochs} epochs.')
        for epoch in range(epochs):
            start_time = time.time()
            trained_model, train_loss, total_memory, mem_reserved = train(device=device, 
                                                                          model=bravo_decoder_model, 
                                                                          loss_fn=loss_of_brain_decoder, 
                                                                          optimizer=optimizer_of_brain_decoder, 
                                                                          train_dataloader=train_dataloader)
            end_time = time.time()
            print(f'Epoch {epoch+1}/{epochs}, {loss_of_brain_decoder.__class__.__name__}: {train_loss:.4f}, Time: {(end_time-start_time)/60:.2f} minutes.')
            print(f'GPU memory usage: {mem_reserved:.2f} GB / {total_memory:.2f} GB.')
        # save the trained model
        torch.save(trained_model.state_dict(), saved_model_path)
    # Test
    elif task == 'test':
        print(f'Testing Brain Decoder.')
        # load the trained model
        bravo_decoder_model.load_state_dict(torch.load(saved_model_path))
        metrics_fn = {
            'MSE' : torch.nn.MSELoss(),
            'MAE' : torch.nn.L1Loss(),
        }
        test(device=device, model=bravo_decoder_model, metrics_fn=metrics_fn, test_dataloader=test_dataloader, saved_test_results_dir_path=saved_test_results_dir_path)
    # Analyze
    elif task == 'analyze':
        analyze_blip_diffusion_embeddings_space(dataloader=train_dataloader, mode='train')
        analyze_blip_diffusion_embeddings_space(dataloader=test_dataloader, mode='test')
        analyze_test_results(saved_test_results_dir_path=saved_test_results_dir_path)
    else:
        raise ValueError(f'Task should be either train or test, but got {task}.')
    
if __name__ == '__main__':
    main()
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
