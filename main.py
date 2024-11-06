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
from models import device, devices_num, num_workers, get_GPU_memory_usage, Image_Decoder, Caption_Decoder
from utils import join_paths, check_and_make_dirs, write_json_file, read_json_file, merge_dicts_if_no_conflict, BLIP_Prior_Tools
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
        if tower_name in ['image', 'i']:
            input_embedding = batches.masked_fmri.to(device)
            target_embedding = batches.blip_image_embedding.to(device)
        elif tower_name in ['text', 't', 'caption', 'c']:
            input_embedding = batches.masked_fmri.to(device)
            # input_embedding = batches.masked_fmri_embedding.to(device)
            target_embedding = batches.blip_caption_embedding_variable.to(device)
        else:
            raise ValueError(f'tower_name={tower_name} is not supported')
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
    
    get_mae_loss = lambda y_pred, y_true: float(np.mean(np.abs(y_pred.flatten()-y_true.flatten())))
    get_mse_loss = lambda y_pred, y_true: float(np.mean((y_pred.flatten()-y_true.flatten())**2))
    get_cosine_similarity = lambda y_pred, y_true: float(np.dot(y_pred, y_true) / (np.linalg.norm(y_pred) * np.linalg.norm(y_true)))
    model.eval()
    maeloss_dict, mseloss_dict, cossimi_dict = {}, {}, {}  
    mem_reserved_list = []
    with torch.no_grad():
        desc = 'Testing' if saved_test_results_dir_path is not None else 'Validating'
        for batches in tqdm(dataloader, desc=desc, leave=True):
            index = batches.index
            if tower_name in ['image', 'i']:
                input_embedding = batches.masked_fmri.to(device)
            elif tower_name in ['text', 't', 'caption', 'c']:
                input_embedding = batches.masked_fmri.to(device)
                # input_embedding = batches.masked_fmri_embedding.to(device)
            # Forward
            pred_embedding  = model(input_embedding)
            index = index.cpu().numpy()
            pred_embedding = pred_embedding.cpu().numpy()
            blip_image_embedding = batches.blip_image_embedding.numpy()
            blip_caption_embedding_fixed = batches.blip_caption_embedding_fixed.numpy()
            blip_caption_embedding_variable = batches.blip_caption_embedding_variable.numpy()
            strings_json_paths = batches.strings_json_path
            image = batches.coco_image.numpy()
            for idx, pred_emb, img, hsi, hsc_fix, hsc_var, strings_json_path in zip(
                                                                        index, pred_embedding, image, 
                                                                        blip_image_embedding, 
                                                                        blip_caption_embedding_fixed, 
                                                                        blip_caption_embedding_variable, 
                                                                        strings_json_paths):
                true_emb = hsi if tower_name in ['image', 'i'] else hsc_var
                save_tag = '_img' if tower_name in ['image', 'i'] else '_cap'
                maeloss_dict[int(idx)] = get_mae_loss(pred_emb, true_emb)
                mseloss_dict[int(idx)] = get_mse_loss(pred_emb, true_emb)
                cossimi_dict[int(idx)] = get_cosine_similarity(pred_emb.flatten(), true_emb.flatten())
                if saved_test_results_dir_path is not None:
                    saved_path = join_paths(saved_test_results_dir_path, str(idx))
                    check_and_make_dirs(saved_path)
                    if tower_name in ['image', 'i']:
                        assert pred_emb.shape == (16, 768), f'pred_emb.shape={pred_emb.shape} != (16, 768)'
                        np.save(join_paths(saved_path, f'bravo{save_tag}'), pred_emb)
                    else:
                        np.save(join_paths(saved_path, f'bravo{save_tag}'), __concat_caption_embedding__(array_fixed=hsc_fix, array_variable=pred_emb))
                    np.save(join_paths(saved_path, f'blip_img.npy'), hsi)
                    np.save(join_paths(saved_path, f'blip_cap.npy'), __concat_caption_embedding__(array_fixed=hsc_fix, array_variable=hsc_var))
                    np.save(join_paths(saved_path, 'coco.npy'), img.astype(np.uint8))
                    assert os.path.exists(strings_json_path), f'strings_json_path={strings_json_path} does not exist!'
                    strings = read_json_file(strings_json_path)
                    write_json_file(path=join_paths(saved_path, 'captions.json'), data={'blip_caption' : strings['category_string']+'. '+strings['blip_caption']})

    # Save the MAE Loss and Cosine Similarity
    avg_maeloss = sum([value for value in maeloss_dict.values()])/len(maeloss_dict)
    avg_mseloss = sum([value for value in mseloss_dict.values()])/len(mseloss_dict)
    avg_cossimi = sum([value for value in cossimi_dict.values()])/len(cossimi_dict)
    max_maeloss_key = max(maeloss_dict, key=maeloss_dict.get)
    min_maeloss_key = min(maeloss_dict, key=maeloss_dict.get)
    max_mseloss_key = max(mseloss_dict, key=mseloss_dict.get)
    min_mseloss_key = min(mseloss_dict, key=mseloss_dict.get)
    max_cossimi_key = max(cossimi_dict, key=cossimi_dict.get)
    min_cossimi_key = min(cossimi_dict, key=cossimi_dict.get)
    print(f'Average MAE loss: {avg_maeloss:.6f}, Max MAE Loss is {max_maeloss_key}: {maeloss_dict[max_maeloss_key]:.6f}, Min MAE Loss is {min_maeloss_key}: {maeloss_dict[min_maeloss_key]:.6f}')
    print(f'Average MSE loss: {avg_mseloss:.6f}, Max MSE Loss is {max_mseloss_key}: {mseloss_dict[max_mseloss_key]:.6f}, Min MSE Loss is {min_mseloss_key}: {mseloss_dict[min_mseloss_key]:.6f}')
    print(f'Average COS simi: {avg_cossimi:.6f}, Max COS Simi is {max_cossimi_key}: {cossimi_dict[max_cossimi_key]:.6f}, Min COS Simi is {min_cossimi_key}: {cossimi_dict[min_cossimi_key]:.6f}')
        
    if saved_test_results_dir_path is not None:
        maeloss_dict = {'max key' : max_maeloss_key, 'max val' : maeloss_dict[max_maeloss_key],
                        'min key' : min_maeloss_key, 'min val' : maeloss_dict[min_maeloss_key], **maeloss_dict}
        mseloss_dict = {'max key' : max_mseloss_key, 'max val' : mseloss_dict[max_mseloss_key],
                        'min key' : min_mseloss_key, 'min val' : mseloss_dict[min_mseloss_key], **mseloss_dict}
        cossimi_dict = {'max key' : max_cossimi_key, 'max val' : cossimi_dict[max_cossimi_key],
                        'min key' : min_cossimi_key, 'min val' : cossimi_dict[min_cossimi_key], **cossimi_dict}
        write_json_file(join_paths(saved_test_results_dir_path, f'{tower_name}_maeloss.json'), maeloss_dict)
        write_json_file(join_paths(saved_test_results_dir_path, f'{tower_name}_mseloss.json'), mseloss_dict)
        write_json_file(join_paths(saved_test_results_dir_path, f'{tower_name}_cossimi.json'), cossimi_dict)

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
    batch_size = configs_dict['train_decoder']['batch_size']
    learning_rate = configs_dict['train_decoder']['learning_rate']
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
    arrays_point = make_nsd_dataset(subj_path=sujb_path, mask_data=mask_data, thresholds=thresholds, labels_string=labels_string)
    uncond_embedding = arrays_point.uncond_embedding
    position_embeddings = arrays_point.position_embeddings
    causal_attention_mask = arrays_point.causal_attention_mask
    null_sample_hidden_states = arrays_point.null_sample_hidden_states
    null_img_embedding, null_cap_embedding = BLIP_Prior_Tools.split_and_concat(null_sample_hidden_states.squeeze())
    regions_saved_dir_path = arrays_point.regions_saved_dir_path

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
        # if tower_name in ['image', 'i']:
        #     # decoder_loss = torch.nn.CrossEntropyLoss()
        #     decoder_loss = Decoder_loss(w1=1, w2=1, w3=0)
        # elif tower_name in ['text', 't', 'caption', 'c']:
        #     decoder_loss = Decoder_loss(w1=1, w2=1, w3=0) 
        # Network
        light_loader = next(iter(test_dataloader))
        
        if tower_name in ['image', 'i']:
            input_shape = light_loader.masked_fmri.shape[1:]
            output_shape = light_loader.blip_image_embedding.shape[1:] 
            decoder_model = Image_Decoder(input_shape=input_shape, output_shape=output_shape)
        elif tower_name in ['text', 't', 'caption', 'c']:
            input_shape = light_loader.masked_fmri.shape[1:]
            # input_shape = light_loader.masked_fmri_embedding.shape[1:]
            output_shape = light_loader.blip_caption_embedding_variable.shape[1:] 
            decoder_model = Caption_Decoder(input_shape=input_shape, output_shape=output_shape)
        print(f'Input Shape  = {input_shape}')
        print(f'Output Shape = {output_shape}')
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
            # lr = learning_rate*((1-epoch/epochs)**0.9)
            # for param_group in optimizer_of_brain_decoder.param_groups:
            #     param_group['lr'] = lr
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
        def __merge_images_with_separators__(
                images_dict : dict[str, Image.Image], saved_dir_path : str,
                separator_width : int = 10, separator_color : tuple[int, int, int] = (255, 255, 255)
        ) -> None:
            names, images = [], []
            for name, image in images_dict.items():
                names.append(name)
                images.append(image)
            name = '__'.join(names)
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
        caption_cossimi = read_json_file(join_paths(saved_test_results_dir_path, 'caption_cossimi.json'))
        image_maeloss   = read_json_file(join_paths(saved_test_results_dir_path, 'image_maeloss.json'))
        image_cossimi   = read_json_file(join_paths(saved_test_results_dir_path, 'image_cossimi.json'))
        for index, dir_path in test_dirs_path_dict.items():
            if index in [0,15,19,118, 208, 600, 641, 661, 1187,1530, 1706]:
                print(f'Generating {index} / {len(test_dirs_path_dict)}')
                coco_matrix = np.load(join_paths(dir_path, 'coco.npy'), allow_pickle=True)
                coco = Image.fromarray(coco_matrix).convert('RGB')
                bravo_img = np.load(join_paths(dir_path, 'bravo_img.npy'), allow_pickle=True)
                bravo_cap = np.load(join_paths(dir_path, 'bravo_cap.npy'), allow_pickle=True)
                blip_img  = np.load(join_paths(dir_path, 'blip_img.npy' ), allow_pickle=True)
                blip_cap  = np.load(join_paths(dir_path, 'blip_cap.npy' ), allow_pickle=True)

                ### test
                # null_img_embedding = np.zeros_like(null_img_embedding)
                # null_cap_embedding = np.zeros_like(null_cap_embedding)
                ### test

                hidden_state_dict = {
                    'blipI+blipC'   : BLIP_Prior_Tools.concatenate_embeddings(img_emb=blip_img , txt_emb=blip_cap),
                    'blipI+bravoC'  : BLIP_Prior_Tools.concatenate_embeddings(img_emb=blip_img , txt_emb=bravo_cap),
                    'bravoI+blipC'  : BLIP_Prior_Tools.concatenate_embeddings(img_emb=bravo_img, txt_emb=blip_cap ),
                    'nullI+nullC'   : BLIP_Prior_Tools.concatenate_embeddings(img_emb=null_img_embedding, txt_emb=null_cap_embedding),
                    'nullI+blipC'   : BLIP_Prior_Tools.concatenate_embeddings(img_emb=null_img_embedding, txt_emb=blip_cap),
                    'blipI+nullC'   : BLIP_Prior_Tools.concatenate_embeddings(img_emb=blip_img, txt_emb=null_cap_embedding),
                    'nullI+bravoC'  : BLIP_Prior_Tools.concatenate_embeddings(img_emb=null_img_embedding, txt_emb=bravo_cap),
                    'bravoI+nullC'  : BLIP_Prior_Tools.concatenate_embeddings(img_emb=bravo_img, txt_emb=null_cap_embedding),
                    'bravoI+bravoC' : BLIP_Prior_Tools.concatenate_embeddings(img_emb=bravo_img, txt_emb=bravo_cap),
                }
                images_dict = {
                    'coco' : coco
                }
                captions_dict = {}
                for key in hidden_state_dict:
                    hidden_state = hidden_state_dict[key]
                    assert hidden_state.shape == position_embeddings.shape, f'hidden_state.shape={hidden_state.shape} != position_embeddings.shape={position_embeddings.shape}'
                    hidden_state += position_embeddings
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
                    start_time = time.time()
                    image = blip2t5_vis_processors['eval'](images_dict[key]).unsqueeze(0).to(device)
                    caption = blip2t5_model.generate({'image' : image, 'prompt' : prompt},
                                                          max_length=100, min_length=30)
                    captions_dict[key] = caption[0]
                    end_time = time.time()
                    print(f'Time taken to generate caption for {key}: {end_time - start_time:.4f} seconds.')

                __merge_images_with_separators__(images_dict=images_dict, saved_dir_path=dir_path)
                captions_json_path = join_paths(dir_path, 'captions.json')
                blip_caption_dict = {'blip_caption' : read_json_file(captions_json_path)['blip_caption']}
                all_captions = merge_dicts_if_no_conflict(dict1=blip_caption_dict, dict2=captions_dict)
                all_captions['caption_maeloss'] = caption_maeloss[str(index)]
                all_captions['caption_cossimi'] = caption_cossimi[str(index)]
                all_captions['image_maeloss']   = image_maeloss[str(index)]
                all_captions['image_cossimi']   = image_cossimi[str(index)]
                write_json_file(captions_json_path, all_captions)
                for k, v in all_captions.items():
                    print(f'{k}: {v}')
                print()

    else:
        raise ValueError(f'Task should be either [train test generate generation], but got {task}.')
    
if __name__ == '__main__':
    main()
    print('Done.\n\n')
