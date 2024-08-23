import os
import re
import time
import torch
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader

from losses import Decoder_loss
from config import configs_dict
from models import device, devices_num, load_blip_models, get_GPU_memory_usage, BraVO_Decoder
from dataset import make_paths_dict, fetch_roi_files_and_labels, NSD_HDF5_Dataset, make_hdf5
from utils import join_paths, check_and_make_dirs
from utils import NSD_saved_dir_path, fmrishape_saved_dir_path, train_results_dir_path, test_results_dir_path


def train(
    device : torch.device,
    model : torch.nn.Module,
    loss_fn : torch.nn.modules.loss._Loss,
    optimizer : torch.optim.Optimizer,
    dataloader : DataLoader,
) -> tuple[torch.nn.Module, float, float, float]:
    """
    """
    model.train()
    torch.set_grad_enabled(True)
    train_loss = []
    mem_reserved_list = []
    for index, masked_embedding, image, canny, multimodal_embedding in tqdm(dataloader, desc='Training', leave=True):
        # Load data to device and set the dtype as float32
        tensors = [index, masked_embedding, image, canny, multimodal_embedding]
        tensors = list(map(lambda t: t.to(device=device, dtype=torch.float32), tensors))
        index, masked_embedding, image, canny, multimodal_embedding = tensors
        # Forward
        pred_embedding  = model(masked_embedding)
        # Compute loss
        # 如果是图像的话，还有风格损失 感知损失 结构损失等
        loss = loss_fn(input=pred_embedding, target=multimodal_embedding)
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
    dataloader : DataLoader,
    saved_test_results_dir_path : str
) -> tuple[float, float, float]:
    """
    """
    model.eval()
    metrics_dict = {} # {key=index, value=MSELoss}
    mem_reserved_list = []
    with torch.no_grad():
        for index, masked_embedding, image, canny, multimodal_embedding in tqdm(dataloader, desc='Testing', leave=True):
            # Load data to device and set the dtype as float32
            tensors = [index, masked_embedding, image, canny, multimodal_embedding]
            tensors = list(map(lambda t: t.to(device=device, dtype=torch.float32), tensors))
            index, masked_embedding, image, canny, multimodal_embedding = tensors
            # Forward
            pred_embedding  = model(masked_embedding)
            # save the results
            index = index.cpu().numpy().astype(np.uint8)
            pred_embedding = pred_embedding.cpu().numpy()
            multimodal_embedding = multimodal_embedding.cpu().numpy()
            image = image.cpu().numpy()
            canny = canny.cpu().numpy()
            for idx, pred, true, img, cny in zip(index, pred_embedding, multimodal_embedding, image, canny):
                metrics_dict[idx] = np.mean(np.power(pred - true, 2))
                saved_path = join_paths(saved_test_results_dir_path, str(idx))
                check_and_make_dirs(saved_path)
                np.save(join_paths(saved_path, 'pred.npy'), pred)
                np.save(join_paths(saved_path, 'true.npy'), true)
                img = Image.fromarray(img.astype(np.uint8))
                img.save(join_paths(saved_path, 'coco_image.png'))
                cny = Image.fromarray(cny.astype(np.uint8))
                cny.save(join_paths(saved_path, 'coco_canny.png'))

    # Monitor GPU memory usage
    total_memory, mem_reserved = get_GPU_memory_usage()
    mem_reserved_list.append(mem_reserved)
    return sum(list(metrics_dict.values()))/len(metrics_dict), total_memory, max(mem_reserved_list)

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
    batch_size = configs_dict['train_decoder']['batch_size'] * devices_num
    learning_rate = configs_dict['train_decoder']['learning_rate']
    epochs = configs_dict['train_decoder']['epochs']
    # roi
    derived_type = configs_dict['NSD_ROIs']['derived_type']
    roi_name = configs_dict['NSD_ROIs']['roi_name']
    thresholds = configs_dict['NSD_ROIs']['thresholds']
    # blip_diffusion
    iter_seed = configs_dict['blip_diffusion']['iter_seed']
    guidance_scale = configs_dict['blip_diffusion']['guidance_scale']
    num_inference_steps = configs_dict['blip_diffusion']['num_inference_steps']
    
    ## Path to save
    # the path of the subj + dataset
    dataset_name = dataset_name.lower()
    subjid_string = f'subj{str(subj_id).zfill(2)}_pairs'
    if dataset_name == 'nsd':
        sujb_path = join_paths(NSD_saved_dir_path, subjid_string)
    elif dataset_name == 'fmri_shape':
        sujb_path = join_paths(fmrishape_saved_dir_path, subjid_string)
    else:
        raise ValueError(f'dataset_name={dataset_name} is not supported.')
    assert os.path.exists(sujb_path), f'dir_path={sujb_path} does not exist.'

    ## Data
    trial_path_dict, rois_path_dict, uncond_embedding = make_paths_dict(subj_path=sujb_path, task=task)
    mask_path_list, labels_string = fetch_roi_files_and_labels(derived_type=derived_type, roi_name=roi_name, thresholds=thresholds, rois_path_dict=rois_path_dict)
    
    ## Path to save
    ## the path of this derived_type, this roi_name and these labels
    saved_subj_train_result_dir_path = join_paths(train_results_dir_path, dataset_name, f'subj{str(subj_id).zfill(2)}', f'{derived_type}_{roi_name}', f'{labels_string}')
    check_and_make_dirs(saved_subj_train_result_dir_path)
    # the path of saving the trained model
    saved_model_path = join_paths(saved_subj_train_result_dir_path, f'ep-{epochs}_bs-{batch_size}_lr-{learning_rate}.pth')
    # path to save the prediected fMRI(whole brain)
    saved_test_results_dir_path = join_paths(test_results_dir_path, dataset_name, f'subj{str(subj_id).zfill(2)}', f'{derived_type}_{roi_name}', f'{labels_string}')
    if os.path.exists(saved_test_results_dir_path):
        shutil.rmtree(saved_subj_train_result_dir_path)
    check_and_make_dirs(saved_test_results_dir_path)
    
    
    ## Algorithm
    # dataloader
    hdf5_path = join_paths(sujb_path, f'preprocessed_{task}_data.hdf5')
    make_hdf5(trial_path_dict=trial_path_dict, mask_path_list=mask_path_list, thresholds=thresholds, hdf5_path=hdf5_path)
    dataloader = DataLoader(dataset=NSD_HDF5_Dataset(hdf5_path=hdf5_path), batch_size=batch_size, shuffle=False, num_workers=6)
    # Network
    light_loader = next(iter(dataloader))
    input_shape  = light_loader[1].shape[1:]  # The shape of masked_embedding
    output_shape = light_loader[-1].shape[1:] # The shape of multimodal_embedding
    bravo_decoder_model = BraVO_Decoder(input_shape=input_shape, output_shape=output_shape)
    print(bravo_decoder_model)
    trainable_parameters = sum(p.numel() for p in bravo_decoder_model.parameters() if p.requires_grad)
    bravo_decoder_model = bravo_decoder_model.to(device=device)
    bravo_decoder_model = torch.nn.DataParallel(bravo_decoder_model)
    print(f'The number of trainable parametes is {trainable_parameters}.')
    # Loss function
    decoder_loss = Decoder_loss(w1=1, w2=0.5)
    # Optimizer
    # optimizer_of_brain_decoder = torch.optim.Adam(bravo_decoder_model.parameters(), lr=learning_rate) 
    optimizer_of_brain_decoder = torch.optim.AdamW(bravo_decoder_model.parameters(), lr=learning_rate) 

    # Train
    if task == 'train':
        print(f'Training Brain Decoder for {epochs} epochs. batch_size={batch_size}, learning_rate={learning_rate}.')
        for epoch in range(epochs):
            start_time = time.time()
            lr = learning_rate*((1-epoch/epochs)**0.9)
            for param_group in optimizer_of_brain_decoder.param_groups:
                param_group['lr'] = lr
            trained_model, train_loss, total_memory, mem_reserved = train(device=device, 
                                                                          model=bravo_decoder_model, 
                                                                          loss_fn=decoder_loss, 
                                                                          optimizer=optimizer_of_brain_decoder, 
                                                                          dataloader=dataloader,
                                                                        )
            end_time = time.time()
            print(f'Epoch {epoch+1}/{epochs}, {decoder_loss.__class__.__name__}: {train_loss:.4f}, Time: {(end_time-start_time)/60:.2f} minutes.')
            print(f'GPU memory usage: {mem_reserved:.2f} GB / {total_memory:.2f} GB.')
            # Save the temporal trained model in each epoch
            torch.save(trained_model.state_dict(), join_paths(saved_subj_train_result_dir_path, f'temporary_ep-{epoch+1}_lr-{learning_rate}.pth'))
        # save the finally trained model, delete the temporal trained model
        for pth_file_path in os.listdir(saved_subj_train_result_dir_path):
            if pth_file_path.startswith('temporary_ep') and pth_file_path.endswith('.pth'):
                os.remove(join_paths(saved_subj_train_result_dir_path, pth_file_path))
        torch.save(trained_model.state_dict(), saved_model_path)
    # Test and Generate
    elif task == 'test':
        # Test
        print(f'Testing Brain Decoder.')
        # load the trained model
        bravo_decoder_model.load_state_dict(torch.load(saved_model_path))
        Euclidean_distance, total_memory, mem_reserved = test(device=device, 
                                          model=bravo_decoder_model, 
                                          dataloader=dataloader, 
                                          saved_test_results_dir_path=saved_test_results_dir_path,
                                    )
        print(f'Averaged MSELoss = {Euclidean_distance:.4f}.')
        print(f'GPU memory usage: {mem_reserved:.2f} GB / {total_memory:.2f} GB.')
        # Generate
        blip_diffusion_model, _, _ = load_blip_models(mode = 'diffusion')
        def __natural_sort_key__(s, _nsre=re.compile('([0-9]+)')):  
            return [int(x) if x.isdigit() else x.lower() for x in _nsre.split(s)]
        for dir_path in sorted(os.listdir(saved_test_results_dir_path), key=__natural_sort_key__):
            print(f'Generating {dir_path} / {len(trial_path_dict)}.')
            pred_multimodal_embedding_path = join_paths(saved_test_results_dir_path, dir_path, 'pred.npy')
            assert os.path.exists(pred_multimodal_embedding_path), f'{pred_multimodal_embedding_path} does not exist.'
            multimodal_embedding = np.load(pred_multimodal_embedding_path, allow_pickle=True)
            assert multimodal_embedding.shape == uncond_embedding.shape
            embedding = np.stack((uncond_embedding, multimodal_embedding), axis=0)
            embedding = torch.from_numpy(embedding).to(device)
            generated_image = blip_diffusion_model.generate_image_via_embedding(
                text_embeddings=embedding,
                seed=iter_seed,
                guidance_scale=guidance_scale,
                height=512,
                width=512,
                num_inference_steps=num_inference_steps,
            )
            generated_image[0].save(join_paths(saved_test_results_dir_path, dir_path, 'output.png'))
    else:
        raise ValueError(f'Task should be either [train test generate generation], but got {task}.')
    
if __name__ == '__main__':
    main()
