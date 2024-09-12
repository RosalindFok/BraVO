import os
import re
import torch
import shutil
import argparse
import platform
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader

from losses import Decoder_loss
from config import configs_dict
from dataset import make_paths_dict, fetch_roi_files_and_labels, NSD_HDF5_Dataset, make_hdf5
from models import device, devices_num, load_blip_models, get_GPU_memory_usage, Image_Decoder, Caption_Decoder
from utils import join_paths, check_and_make_dirs, NSD_saved_dir_path, fmrishape_saved_dir_path, train_results_dir_path, test_results_dir_path


def train(
    device : torch.device,
    model : torch.nn.Module,
    loss_fn : torch.nn.modules.loss._Loss,
    optimizer : torch.optim.Optimizer,
    dataloader : DataLoader,
    tower_name : str
) -> tuple[torch.nn.Module, float, float, float]:
    """
    """
    model.train()
    torch.set_grad_enabled(True)
    torch.cuda.empty_cache()
    train_loss = []
    mem_reserved_list = []
    tower_name = tower_name.lower()
    for batches in tqdm(dataloader, desc='Training', leave=True):
        # select the tower, load data to device and set the dtype as float32
        if tower_name in ['image', 'i']:
            masked_embedding = batches.masked_embedding_image.to(device)
            hidden_states = batches.hidden_states_image.to(device)
        elif tower_name in ['text', 't', 'caption', 'c']:
            masked_embedding = batches.masked_embedding_caption.to(device)
            hidden_states = batches.hidden_states_caption.to(device)
        else:
            raise ValueError(f'tower_name={tower_name} is not supported')
        # Forward
        pred_embedding  = model(masked_embedding)
        # Compute loss
        loss = loss_fn(input=pred_embedding, target=hidden_states)
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
    tower_name : str,
    saved_test_results_dir_path : str = None
) -> tuple[float, float, float]:
    """
    """
    model.eval()
    mseloss_list = []
    mem_reserved_list = []
    with torch.no_grad():
        desc = 'Testing' if saved_test_results_dir_path is not None else 'Validating'
        for batches in tqdm(dataloader, desc=desc, leave=True):
            index = batches.index
            # select the tower, load data to device and set the dtype as float32
            if tower_name in ['image', 'i']:
                masked_embedding = batches.masked_embedding_image.to(device)
                target_embedding = batches.hidden_states_image.to(device)
            elif tower_name in ['text', 't', 'caption', 'c']:
                masked_embedding = batches.masked_embedding_caption.to(device)
                target_embedding = batches.hidden_states_caption.to(device)
            else:
                raise ValueError(f'tower_name={tower_name} is not supported')
            # Forward
            pred_embedding  = model(masked_embedding)
            mseloss = torch.nn.MSELoss()(pred_embedding, target_embedding).item()
            mseloss_list.append(mseloss)
            # save the results
            if saved_test_results_dir_path is not None:
                index = index.cpu().numpy()
                pred_embedding = pred_embedding.cpu().numpy()
                hidden_states_image = batches.hidden_states_image.numpy()
                hidden_states_caption = batches.hidden_states_caption.numpy()
                image = batches.image.numpy()
                for idx, pred_emb, true_img_emb, true_cap_emb, img in zip(index, pred_embedding, hidden_states_image, hidden_states_caption, image):
                    saved_path = join_paths(saved_test_results_dir_path, str(idx))
                    check_and_make_dirs(saved_path)
                    if tower_name in ['image', 'i']:
                        np.save(join_paths(saved_path, 'bravo_img.npy'), pred_emb)
                    elif tower_name in ['text', 't', 'caption', 'c']:
                        np.save(join_paths(saved_path, 'bravo_cap.npy'), pred_emb)
                    np.save(join_paths(saved_path, 'blip_img.npy'), true_img_emb)
                    np.save(join_paths(saved_path, 'blip_cap.npy'), true_cap_emb)
                    np.save(join_paths(saved_path, 'coco.npy'), img.astype(np.uint8))

    # Monitor GPU memory usage
    total_memory, mem_reserved = get_GPU_memory_usage()
    mem_reserved_list.append(mem_reserved)
    return sum(mseloss_list)/len(mseloss_list), total_memory, max(mem_reserved_list)

def main() -> None:
    ## Task
    parser = argparse.ArgumentParser(description='Select from train or test.')
    parser.add_argument('--task', type=str, help='task: t or g.')
    parser.add_argument('--tower_name', type=str, default='image', help='tower_name: image or caption.')
    args = parser.parse_args()
    task = args.task.lower()
    tower_name = args.tower_name.lower()

    ## Hyperparameters
    # subj id
    subj_id = configs_dict['subj_id']
    # dataset name
    dataset_name = configs_dict['dataset_name']
    # train brain decoder
    batch_size = configs_dict['train_decoder']['batch_size'] * devices_num
    batch_size = batch_size * 16 if tower_name in ['image', 'i'] else batch_size
    learning_rate = configs_dict['train_decoder']['learning_rate']
    # learning_rate = learning_rate * 0.1 if tower_name in ['image', 'i'] else learning_rate
    epochs = configs_dict['train_decoder']['epochs']
    # roi
    derived_type = configs_dict['NSD_ROIs']['derived_type']
    roi_name = configs_dict['NSD_ROIs']['roi_name']
    if tower_name in ['image', 'i']:
        thresholds = configs_dict['NSD_ROIs']['thresholds']['primary_visual_cortex']
    elif tower_name in ['text', 't', 'caption', 'c']:
        thresholds = configs_dict['NSD_ROIs']['thresholds']['higher_visual_cortex']
    else:
        raise ValueError(f'tower_name={tower_name} is not supported.')
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
    train_trial_path_dict, test_trial_path_dict, rois_path_dict, uncond_embedding, causal_attention_mask = make_paths_dict(subj_path=sujb_path)
    mask_path_list, labels_string = fetch_roi_files_and_labels(derived_type=derived_type, roi_name=roi_name, 
                                                               thresholds=thresholds, rois_path_dict=rois_path_dict
                                                            )
    
    ## Path to save
    # the path of training results
    path_info = (dataset_name, f'subj{str(subj_id).zfill(2)}', f'{derived_type}_{roi_name}', f'{labels_string}')
    saved_subj_train_result_dir_path = join_paths(train_results_dir_path, *path_info)
    check_and_make_dirs(saved_subj_train_result_dir_path)
    saved_model_path = join_paths(saved_subj_train_result_dir_path, f'tower-{tower_name}_ep-{epochs}_lr-{learning_rate}.pth')
    # the path of testing results
    saved_test_results_dir_path = join_paths(test_results_dir_path, *path_info[:-1])
    check_and_make_dirs(saved_test_results_dir_path)
    
    # Train-Valid and Test
    if task == 't':
        # HDF5 files
        hdf5_dirs = join_paths(sujb_path, f'hdf5-{labels_string}')
        check_and_make_dirs(hdf5_dirs)
        train_hdf5_path = join_paths(hdf5_dirs, f'train_data.hdf5')
        test_hdf5_path = join_paths(hdf5_dirs, f'test_data.hdf5')
        train_temp_dir_path = join_paths(hdf5_dirs, 'train_temp')
        check_and_make_dirs(train_temp_dir_path)
        test_temp_dir_path = join_paths(hdf5_dirs, 'test_temp')
        check_and_make_dirs(test_temp_dir_path)
        for hdf5_path, trial_path_dict, temp_dir_path in zip([train_hdf5_path      , test_hdf5_path      ], 
                                                             [train_trial_path_dict, test_trial_path_dict],
                                                             [train_temp_dir_path  , test_temp_dir_path  ]):
            make_hdf5(trial_path_dict=trial_path_dict, mask_path_list=mask_path_list, thresholds=thresholds, 
                      hdf5_path=hdf5_path, temp_dir_path=temp_dir_path
                    )
        # dataloader
        num_workers = 6 if platform.system() == 'Linux' else 0
        train_dataloader = DataLoader(dataset=NSD_HDF5_Dataset(hdf5_path=train_hdf5_path), batch_size=batch_size, 
                                      shuffle=False, num_workers=num_workers)
        test_dataloader  = DataLoader(dataset=NSD_HDF5_Dataset(hdf5_path=test_hdf5_path),  batch_size=batch_size, 
                                      shuffle=False, num_workers=num_workers)
        # Loss function
        decoder_loss = Decoder_loss(w1=1, w2=0.5) 
        # Network
        light_loader = next(iter(test_dataloader))
        if tower_name in ['image', 'i']:
            input_shape  = light_loader.masked_embedding_image.shape[1:] 
            output_shape = light_loader.hidden_states_image.shape[1:] 
            decoder_model = Image_Decoder(input_shape=input_shape, output_shape=output_shape)
        elif tower_name in ['text', 't', 'caption', 'c']:
            input_shape  = light_loader.masked_embedding_caption.shape[1:] 
            output_shape = light_loader.hidden_states_caption.shape[1:] 
            decoder_model = Caption_Decoder(input_shape=input_shape, output_shape=output_shape)
        trainable_parameters = sum(p.numel() for p in decoder_model.parameters() if p.requires_grad)
        decoder_model = decoder_model.to(device=device)
        # decoder_model = torch.nn.DataParallel(decoder_model)
        print(f'The number of trainable parametes of {decoder_model.__class__.__name__} is {trainable_parameters}.')
        # Optimizer
        optimizer_of_brain_decoder = torch.optim.AdamW(decoder_model.parameters(), lr=learning_rate) 
        print(f'Training Brain {decoder_model.__class__.__name__} for {epochs} epochs. batch_size={batch_size}, learning_rate={learning_rate}.')
        valid_mseloss_list = []
        for epoch in range(epochs):
            print(f'Tower {tower_name}, Epoch {epoch+1}/{epochs}')
            # train
            lr = learning_rate*((1-epoch/epochs)**0.9)
            for param_group in optimizer_of_brain_decoder.param_groups:
                param_group['lr'] = lr
            trained_model, train_loss, total_memory, mem_reserved = train(device=device, 
                                                                          model=decoder_model, 
                                                                          loss_fn=decoder_loss, 
                                                                          optimizer=optimizer_of_brain_decoder, 
                                                                          dataloader=train_dataloader,
                                                                          tower_name=tower_name
                                                                        )
            # Save the temporal trained model in each epoch
            temporary_model_path = join_paths(saved_subj_train_result_dir_path, f'temporary_ep-{epoch+1}_lr-{learning_rate}.pth')
            torch.save(trained_model.state_dict(), temporary_model_path)
            # valid
            decoder_model.load_state_dict(torch.load(temporary_model_path, weights_only=True))
            MSELoss, _, _ = test(device=device, 
                                 model=decoder_model, 
                                 dataloader=test_dataloader,
                                 tower_name=tower_name, 
                                 saved_test_results_dir_path=None
                                )
            print(f'GPU memory usage: {mem_reserved:.2f} GB / {total_memory:.2f} GB.')
            print(f'Train {decoder_loss.__class__.__name__} = {train_loss:.6f}, Valid MSELoss = {MSELoss:.6f}.\n')
            valid_mseloss_list.append(MSELoss)

        # save the finally trained model, delete the temporal trained model
        for pth_file_path in os.listdir(saved_subj_train_result_dir_path):
            if pth_file_path.startswith('temporary_ep') and pth_file_path.endswith('.pth'):
                os.remove(join_paths(saved_subj_train_result_dir_path, pth_file_path))
        torch.save(trained_model.state_dict(), saved_model_path)

        # Draw a line chart for valid_mseloss_list
        plt.plot(range(1, epochs+1), valid_mseloss_list)
        plt.xlabel('Epoch')
        plt.ylabel('MSELoss')
        plt.title(f'Brain Decoder MSELoss (valid)')
        plt.savefig(join_paths(saved_subj_train_result_dir_path, f'brain_{tower_name}_decoder_valid_mseloss.png'))
        plt.close()

        # test
        print(f'Testing Brain Decoder.')
        # load the trained model
        decoder_model.load_state_dict(torch.load(saved_model_path, weights_only=True))
        MSELoss, total_memory, mem_reserved = test(device=device, 
                                      model=decoder_model, 
                                      dataloader=test_dataloader, 
                                      tower_name=tower_name,
                                      saved_test_results_dir_path=saved_test_results_dir_path
                                    )
        print(f'Averaged MSELoss = {MSELoss:.4f}.')
        print(f'GPU memory usage: {mem_reserved:.2f} GB / {total_memory:.2f} GB.')
    
    # Generate
    elif task == 'g':   
        def __natural_sort_key__(s, _nsre=re.compile('([0-9]+)')):  
            return [int(x) if x.isdigit() else x.lower() for x in _nsre.split(s)]
        
        def __concatenate_embeddings__(img_emb : np.ndarray, txt_emb : np.ndarray) -> np.ndarray:
            assert img_emb.shape == (16, 768), f'img_emb={img_emb.shape} should be (16, 768)'
            assert txt_emb.shape == (61, 768), f'txt_emb={txt_emb.shape} should be (61, 768)'
            result = np.concatenate((txt_emb[:2, :], img_emb, txt_emb[2:, :]), axis=0)
            assert result.shape == (77, 768), f'result.shape={result.shape} should be (77, 768)'
            return result
        
        def __merge_images_with_separators__(
                images_dict : dict[str, Image.Image], saved_dir_path : str,
                separator_width : int = 10, separator_color : tuple[int, int, int] = (255, 255, 255)
        ) -> None:
            names, images = [], []
            for name, image in images_dict.items():
                names.append(name)
                images.append(image)
            name = '_'.join(names)
            total_width = sum(image.width for image in images) + separator_width * (len(images) - 1)  
            max_height = max(image.height for image in images)  
            new_img = Image.new('RGB', (total_width, max_height), separator_color)  
            current_x = 0  
            for img in images:  
                new_img.paste(img, (current_x, 0))  
                current_x += img.width + separator_width  
            new_img.save(join_paths(saved_dir_path, f'{name}.png'))
    
        blip_diffusion_model, _, _ = load_blip_models(mode = 'diffusion')
        uncond_embedding = torch.from_numpy(uncond_embedding).to(device)
        causal_attention_mask = torch.from_numpy(causal_attention_mask).to(device)
        test_dirs_path_list = sorted(os.listdir(saved_test_results_dir_path), key=__natural_sort_key__)
        for dir_path in test_dirs_path_list:
            print(f'Generating {dir_path} / {len(test_dirs_path_list)}.')
            dir_path = join_paths(saved_test_results_dir_path, dir_path)
            assert (num_files := len(contents := os.listdir(dir_path))) in [5, 6], f'Unexpected number of files: {num_files}. Contents: {contents}'
            coco_matrix = np.load(join_paths(dir_path, 'coco.npy'), allow_pickle=True)
            coco = Image.fromarray(coco_matrix).convert('RGB')
            bravo_img = np.load(join_paths(dir_path, 'bravo_img.npy'), allow_pickle=True)
            bravo_cap = np.load(join_paths(dir_path, 'bravo_cap.npy'), allow_pickle=True)
            blip_img  = np.load(join_paths(dir_path, 'blip_img.npy' ), allow_pickle=True)
            blip_cap  = np.load(join_paths(dir_path, 'blip_cap.npy' ), allow_pickle=True)
            hidden_state_dict = {
                'blip'    : __concatenate_embeddings__(img_emb=blip_img , txt_emb=blip_cap ),
                'caption' : __concatenate_embeddings__(img_emb=blip_img , txt_emb=bravo_cap),
                'image'   : __concatenate_embeddings__(img_emb=bravo_img, txt_emb=blip_cap ),
                'img+cap' : __concatenate_embeddings__(img_emb=bravo_img, txt_emb=bravo_cap)
            }
            images_dict = {
                'coco' : coco
            }
            for key in hidden_state_dict:
                hidden_state = hidden_state_dict[key]
                hidden_state = torch.from_numpy(hidden_state).unsqueeze(0).to(device)
                assert hidden_state.shape == uncond_embedding.shape, f'{hidden_state.shape} != {uncond_embedding.shape}'
                generated_image = blip_diffusion_model.generate_image_via_embedding(
                                                        uncond_embedding=uncond_embedding,
                                                        hidden_states=hidden_state,
                                                        causal_attention_mask=causal_attention_mask,
                                                        seed=iter_seed,
                                                        guidance_scale=guidance_scale,
                                                        height=coco_matrix.shape[0],
                                                        width=coco_matrix.shape[1],
                                                        num_inference_steps=num_inference_steps,
                                                    )
                images_dict[key] = generated_image.convert('RGB')
            
            __merge_images_with_separators__(images_dict=images_dict, saved_dir_path=dir_path)

    else:
        raise ValueError(f'Task should be either [train test generate generation], but got {task}.')
    
if __name__ == '__main__':
    main()
    print('Done.\n\n')
