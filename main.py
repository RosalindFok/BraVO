import os
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from collections import namedtuple
from torch.utils.data import DataLoader

from losses import Decoder_loss
from config import configs_dict
from dataset import fetch_nsd_rois_and_labels, NSD_Dataset, make_nsd_dataset
from models import device, devices_num, num_workers, get_GPU_memory_usage, Image_Decoder, Caption_Decoder, fMRI2Image
from utils import join_paths, check_and_make_dirs, write_json_file, read_json_file, merge_dicts_if_no_conflict
from utils import train_results_dir_path, test_results_dir_path, nsd_subject_saved_dir_path, fmrishape_subject_saved_dir_path


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
        # if tower_name in ['image', 'i']:
        #     input_embedding = batches.blip_masked_embedding.to(device)
        #     target_embedding = batches.hidden_states_image.to(device)
        # elif tower_name in ['text', 't', 'caption', 'c']:
        #     input_embedding = batches.blip_masked_embedding.to(device)
        #     target_embedding = batches.hidden_states_caption_variable.to(device)
        # else:
        #     raise ValueError(f'tower_name={tower_name} is not supported')

        ### test
        input_embedding = batches.original_masked_fmri.to(device)
        target_embedding = batches.image.to(device)
        ### test

        # Forward
        pred_embedding  = model(input_embedding)
        # Compute loss
        loss = loss_fn(input=pred_embedding, target=target_embedding)
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
) -> tuple[float, float]:
    """
    """
    def __concat_caption_embedding__(array_fixed : np.ndarray, array_variable : np.ndarray) -> np.ndarray:
        assert array_fixed.shape == (3, 768)
        assert array_variable.shape == (58, 768)
        array = np.concatenate((array_fixed[:-1], array_variable, array_fixed[-1].reshape(1, -1)), axis=0)
        assert array.shape == (61, 768)
        return array
    
    get_mae_loss = lambda y_pred, y_true: float(np.mean(np.abs(y_pred-y_true)))
    get_mse_loss = lambda y_pred, y_true: float(np.mean((y_pred-y_true)**2))
    model.eval()
    maeloss_dict, mseloss_dict = {}, {}
    mem_reserved_list = []
    with torch.no_grad():
        desc = 'Testing' if saved_test_results_dir_path is not None else 'Validating'
        for batches in tqdm(dataloader, desc=desc, leave=True):
            index = batches.index
            # # select the tower, load data to device and set the dtype as float32
            # if tower_name in ['image', 'i']:
            #     input_embedding = batches.blip_masked_embedding.to(device)
            # elif tower_name in ['text', 't', 'caption', 'c']:
            #     input_embedding = batches.blip_masked_embedding.to(device)
            # else:
            #     raise ValueError(f'tower_name={tower_name} is not supported')

            ### test
            input_embedding = batches.original_masked_fmri.to(device)
            ### test

            # Forward
            pred_embedding  = model(input_embedding)
            index = index.cpu().numpy()
            pred_embedding = pred_embedding.cpu().numpy()

            ### test
            image = batches.image.cpu().numpy()
            for idx, pred_emb, img in zip(index, pred_embedding, image):
                maeloss_dict[int(idx)] = get_mae_loss(pred_emb, img)
                mseloss_dict[int(idx)] = get_mse_loss(pred_emb, img)
                if saved_test_results_dir_path is not None:
                    saved_path = join_paths(saved_test_results_dir_path, str(idx))
                    check_and_make_dirs(saved_path)
                    pred_emb =  (((pred_emb-pred_emb.min())/(pred_emb.max()-pred_emb.min())) * 255).astype(np.uint8)
                    img = (img * 255).astype(np.uint8)
                    Image.fromarray(pred_emb).save(join_paths(saved_path, 'prediction.png'))
                    Image.fromarray(img).save(join_paths(saved_path, 'groundtruth.png'))
            ### test

            # hidden_states_image = batches.hidden_states_image.numpy()
            # hidden_states_caption_fixed = batches.hidden_states_caption_fixed.numpy()
            # hidden_states_caption_variable = batches.hidden_states_caption_variable.numpy()
            # strings_json_paths = batches.strings_json_path
            # image = batches.image.numpy()
            # for idx, pred_emb, img, hsi, hsc_fix, hsc_var, strings_json_path in zip(
            #                                                             index, pred_embedding, image, 
            #                                                             hidden_states_image, 
            #                                                             hidden_states_caption_fixed, 
            #                                                             hidden_states_caption_variable, 
            #                                                             strings_json_paths):
            #     true_emb = hsi if tower_name in ['image', 'i'] else hsc_var
            #     save_tag = '_img' if tower_name in ['image', 'i'] else '_cap'
            #     maeloss_dict[int(idx)] = get_mae_loss(pred_emb, true_emb)
            #     mseloss_dict[int(idx)] = get_mse_loss(pred_emb, true_emb)
            #     if saved_test_results_dir_path is not None:
            #         saved_path = join_paths(saved_test_results_dir_path, str(idx))
            #         check_and_make_dirs(saved_path)
            #         np.save(join_paths(saved_path, f'bravo{save_tag}'), __concat_caption_embedding__(array_fixed=hsc_fix, array_variable=pred_emb))
            #         np.save(join_paths(saved_path, f'blip_img.npy'), hsi)
            #         np.save(join_paths(saved_path, f'blip_cap.npy'), __concat_caption_embedding__(array_fixed=hsc_fix, array_variable=true_emb))
            #         np.save(join_paths(saved_path, 'coco.npy'), img.astype(np.uint8))
            #         strings = read_json_file(strings_json_path)
            #         write_json_file(path=join_paths(saved_path, 'captions.json'), data={'blip' : strings['category_string'] + strings['blip_caption'] })

    # Save the MAE and MSE loss
    avg_maeloss = sum([value for value in maeloss_dict.values()])/len(maeloss_dict)
    avg_mseloss = sum([value for value in mseloss_dict.values()])/len(mseloss_dict)
    max_maeloss_key = max(maeloss_dict, key=maeloss_dict.get)
    min_maeloss_key = min(maeloss_dict, key=maeloss_dict.get)
    max_mseloss_key = max(mseloss_dict, key=mseloss_dict.get)
    min_mseloss_key = min(mseloss_dict, key=mseloss_dict.get)
    print(f'Average MAE loss: {avg_maeloss:.6f}, Max MAELoss is {max_maeloss_key}: {maeloss_dict[max_maeloss_key]:.6f}, Min MAELoss is {min_maeloss_key}: {maeloss_dict[min_maeloss_key]:.6f}')
    print(f'Average MSE loss: {avg_mseloss:.6f}, Max MSELoss is {max_mseloss_key}: {mseloss_dict[max_mseloss_key]:.6f}, Min MSELoss is {min_mseloss_key}: {mseloss_dict[min_mseloss_key]:.6f}')
        
    if saved_test_results_dir_path is not None:
        maeloss_dict = {'max key' : max_maeloss_key, 'max val' : maeloss_dict[max_maeloss_key],
                        'min key' : min_maeloss_key, 'min val' : maeloss_dict[min_maeloss_key], **maeloss_dict}
        mseloss_dict = {'max key' : max_mseloss_key, 'max val' : mseloss_dict[max_mseloss_key],
                        'min key' : min_mseloss_key, 'min val' : mseloss_dict[min_mseloss_key], **mseloss_dict}
        write_json_file(join_paths(saved_test_results_dir_path, f'{tower_name}_maeloss.json'), maeloss_dict)
        write_json_file(join_paths(saved_test_results_dir_path, f'{tower_name}_mseloss.json'), mseloss_dict)

    # Monitor GPU memory usage
    total_memory, mem_reserved = get_GPU_memory_usage()
    mem_reserved_list.append(mem_reserved)
    return total_memory, max(mem_reserved_list)

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
    # batch_size = batch_size * 16 if tower_name in ['image', 'i'] else batch_size
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
    if dataset_name == 'nsd':
        sujb_path = nsd_subject_saved_dir_path
    elif dataset_name == 'fmri_shape':
        sujb_path = fmrishape_subject_saved_dir_path
    else:
        raise ValueError(f'dataset_name={dataset_name} is not supported.')
    assert os.path.exists(sujb_path), f'dir_path={sujb_path} does not exist.'

    ## Data
    rois_setup = namedtuple('rois_setup', ['derived_type', 'roi_name', 'thresholds'])
    rois_setup = rois_setup(derived_type, roi_name, thresholds)
    mask_data, thresholds, labels_string = fetch_nsd_rois_and_labels(subj_path=sujb_path, rois_setup=rois_setup)
    uncond_embedding_path, causal_attention_mask_path, regions_saved_dir_path = make_nsd_dataset(subj_path=sujb_path, mask_data=mask_data, thresholds=thresholds, labels_string=labels_string)
    uncond_embedding = np.load(uncond_embedding_path, allow_pickle=True)
    assert uncond_embedding.shape == (1, 77, 768), f'uncond_embedding.shape={uncond_embedding.shape} != (1, 77, 768)'
    causal_attention_mask = np.load(causal_attention_mask_path, allow_pickle=True)
    assert causal_attention_mask.shape == (1, 1, 77, 77), f'causal_attention_mask.shape={causal_attention_mask.shape} != (1, 1, 77, 77)'
    
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
        # dataloader
        train_dataloader = DataLoader(dataset=NSD_Dataset(join_paths(regions_saved_dir_path, 'train')), 
                                      batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_dataloader  = DataLoader(dataset=NSD_Dataset(join_paths(regions_saved_dir_path, 'test')),  
                                      batch_size=batch_size, shuffle=False, num_workers=num_workers)
        # Loss function
        decoder_loss = Decoder_loss(w1=1, w2=1, w3=0) 
        # Network
        light_loader = next(iter(test_dataloader))
        
        ### test
        input_shape = light_loader.original_masked_fmri.shape[1:] 
        output_shape = light_loader.image.shape[1:] 
        decoder_model = fMRI2Image(input_shape=input_shape, output_shape=output_shape)
        ### test

        # if tower_name in ['image', 'i']:
        #     input_shape  = light_loader.blip_masked_embedding.shape[1:] 
        #     output_shape = light_loader.hidden_states_image.shape[1:] 
        #     decoder_model = Image_Decoder(input_shape=input_shape, output_shape=output_shape)
        # elif tower_name in ['text', 't', 'caption', 'c']:
        #     input_shape  = light_loader.blip_masked_embedding.shape[1:] 
        #     output_shape = light_loader.hidden_states_caption_variable.shape[1:] 
        #     decoder_model = Caption_Decoder(input_shape=input_shape, output_shape=output_shape)
        print(f'Input Shape = {input_shape}, Output Shape = {output_shape}')
        trainable_parameters = sum(p.numel() for p in decoder_model.parameters() if p.requires_grad)
        decoder_model = decoder_model.to(device=device)
        print(decoder_model)
        # decoder_model = torch.nn.DataParallel(decoder_model)
        print(f'The number of trainable parametes of {decoder_model.__class__.__name__} is {trainable_parameters}.')
        # Optimizer
        optimizer_of_brain_decoder = torch.optim.Adam(decoder_model.parameters(), lr=learning_rate) 
        # optimizer_of_brain_decoder = torch.optim.AdamW(decoder_model.parameters(), lr=learning_rate) 
        # optimizer_of_brain_decoder = torch.optim.SGD(decoder_model.parameters(), lr=learning_rate, momentum=0.9)
        # optimizer_of_brain_decoder = torch.optim.RMSprop(decoder_model.parameters(), lr=learning_rate)
        print(f'Training Brain {decoder_model.__class__.__name__} for {epochs} epochs. batch_size={batch_size}, learning_rate={learning_rate}.')
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
            print(f'Train {decoder_loss.__class__.__name__} = {train_loss:.6f}')
            # valid
            decoder_model.load_state_dict(torch.load(temporary_model_path, weights_only=True))
            _, _ = test(device=device, 
                        model=decoder_model, 
                        dataloader=test_dataloader,
                        tower_name=tower_name, 
                        saved_test_results_dir_path=None
                    )
            # GPU memory usage
            print(f'GPU memory usage: {mem_reserved:.2f} GB / {total_memory:.2f} GB.')

        # save the finally trained model, delete the temporal trained model
        for pth_file_path in os.listdir(saved_subj_train_result_dir_path):
            if pth_file_path.startswith('temporary_ep') and pth_file_path.endswith('.pth'):
                os.remove(join_paths(saved_subj_train_result_dir_path, pth_file_path))
        torch.save(trained_model.state_dict(), saved_model_path)

        # test
        print(f'Testing Brain Decoder.')
        # load the trained model
        decoder_model.load_state_dict(torch.load(saved_model_path, weights_only=True))
        total_memory, mem_reserved = test(device=device, 
                                          model=decoder_model, 
                                          dataloader=test_dataloader, 
                                          tower_name=tower_name,
                                          saved_test_results_dir_path=saved_test_results_dir_path
                                    )
        print(f'GPU memory usage: {mem_reserved:.2f} GB / {total_memory:.2f} GB.')
    
    # Generate
    elif task == 'g':   
        from models import load_blip_models
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
        blip2t5_model, blip2t5_vis_processors, _ = load_blip_models(mode='caption') 
        prompt = configs_dict['blip_caption']['prompt']
        uncond_embedding = torch.from_numpy(uncond_embedding).to(device)
        causal_attention_mask = torch.from_numpy(causal_attention_mask).to(device)
        test_dirs_path_dict = {int(path):join_paths(saved_test_results_dir_path, path) for path in os.listdir(saved_test_results_dir_path) if os.path.isdir(join_paths(saved_test_results_dir_path, path))}
        sorted_keys = sorted(test_dirs_path_dict.keys())
        test_dirs_path_dict = {key:test_dirs_path_dict[key] for key in sorted_keys}
        caption_maeloss = read_json_file(join_paths(saved_test_results_dir_path, 'caption_maeloss.json'))
        caption_mseloss = read_json_file(join_paths(saved_test_results_dir_path, 'caption_mseloss.json'))
        # image_maeloss   = read_json_file(join_paths(saved_test_results_dir_path, 'image_maeloss.json'))
        # image_mseloss   = read_json_file(join_paths(saved_test_results_dir_path, 'image_mseloss.json'))
        for index, dir_path in test_dirs_path_dict.items():
            print(f'Generating {index} / {len(test_dirs_path_dict)}')
            coco_matrix = np.load(join_paths(dir_path, 'coco.npy'), allow_pickle=True)
            coco = Image.fromarray(coco_matrix).convert('RGB')
            # bravo_img = np.load(join_paths(dir_path, 'bravo_img.npy'), allow_pickle=True)
            bravo_cap = np.load(join_paths(dir_path, 'bravo_cap.npy'), allow_pickle=True)
            blip_img  = np.load(join_paths(dir_path, 'blip_img.npy' ), allow_pickle=True)
            blip_cap  = np.load(join_paths(dir_path, 'blip_cap.npy' ), allow_pickle=True)

            hidden_state_dict = {
                'blip'    : __concatenate_embeddings__(img_emb=blip_img , txt_emb=blip_cap),
                'bravo' : __concatenate_embeddings__(img_emb=blip_img , txt_emb=bravo_cap),
                # 'image'   : __concatenate_embeddings__(img_emb=bravo_img, txt_emb=blip_cap ),
                # 'img+cap' : __concatenate_embeddings__(img_emb=bravo_img, txt_emb=bravo_cap)
            }
            images_dict = {
                'coco' : coco
            }
            captions_dict = {}
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
                if not key == 'blip':
                    start_time = time.time()
                    image = blip2t5_vis_processors['eval'](images_dict[key]).unsqueeze(0).to(device)
                    caption = blip2t5_model.generate({'image' : image, 'prompt' : prompt},
                                                          max_length=100, min_length=30)
                    captions_dict[key] = caption[0]
                    end_time = time.time()
                    print(f'Time taken to generate caption for {key}: {end_time - start_time:.4f} seconds.')
            
            __merge_images_with_separators__(images_dict=images_dict, saved_dir_path=dir_path)
            captions_json_path = join_paths(dir_path, 'captions.json')
            blip_caption_dict = {'blip' : read_json_file(captions_json_path)['blip']}
            all_captions = merge_dicts_if_no_conflict(dict1=blip_caption_dict, dict2=captions_dict)
            all_captions['caption_maeloss'] = caption_maeloss[str(index)]
            all_captions['caption_mseloss'] = caption_mseloss[str(index)]
            write_json_file(captions_json_path, all_captions)
            for k, v in all_captions.items():
                print(f'{k}: {v}')
            print()
    else:
        raise ValueError(f'Task should be either [train test generate generation], but got {task}.')
    
if __name__ == '__main__':
    main()
    print('Done.\n\n')
   
