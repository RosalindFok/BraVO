import os
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from datetime import datetime
from collections import namedtuple
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from losses import Decoder_loss
from config import configs_dict
from dataset import fetch_nsd_rois_and_labels, mask_fMRI, NSD_Dataset
from models import device, devices_num, num_workers, get_GPU_memory_usage, load_blip_models, BraVO_Decoder
from utils import join_paths, write_json_file, read_json_file, get_items_in_list_via_substrs
from utils import train_results_dir_path, test_results_dir_path, nsd_subject_saved_dir_path, run_files_path, fmrishape_subject_saved_dir_path

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
    train_loss = []
    mem_reserved_list = []
    for batches in tqdm(dataloader, desc='Training', leave=True):
        image = batches.image.to(device)
        masked_fmri = batches.masked_fmri.to(device)
        image_embedding = batches.image_embedding.to(device)
        # forward
        pred_image_embedding = model(masked_fmri)
        # compute loss
        loss = loss_fn(pred_image_embedding.view(-1, 4), image_embedding.view(-1, 4))
        train_loss.append(loss.item())
        assert not torch.isnan(loss), 'loss is nan, stop training!'
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
    # Tools of computing embedding similarity
    get_mae_loss = lambda y_pred, y_true: float(np.mean(np.abs(y_pred.flatten()-y_true.flatten())))
    get_mse_loss = lambda y_pred, y_true: float(np.mean((y_pred.flatten()-y_true.flatten())**2))
    get_accuracy = lambda y_pred, y_true: np.sum(y_pred.flatten()==y_true.flatten())/y_pred.flatten().shape[0]
    maeloss_dict, mseloss_dict = {}, {}
    
    # create saved_path for each sample in test set
    def __make_saved_path__(trial : str) -> str:
        saved_path = join_paths(saved_test_results_dir_path, trial)
        os.makedirs(saved_path, exist_ok=True)
        return saved_path
    
    model.eval()
    mem_reserved_list = []
    acc_list = []
    with torch.no_grad():
        desc = 'Testing' if saved_test_results_dir_path is not None else 'Validating'
        for batches in tqdm(dataloader, desc=desc, leave=True):
            trials = batches.trial
            images = batches.image.cpu().numpy()
            masked_fmri = batches.masked_fmri.to(device)
            pred_image_embedding  = model(masked_fmri).cpu().numpy()
            image_embedding = batches.image_embedding.cpu().numpy()
            # 
            for trial, image, pred, true in zip(trials, images, pred_image_embedding, image_embedding):
                # pred = pred.flatten()
                # true = true.flatten()
                # maeloss_dict[trial] = get_mae_loss(pred, true)
                # mseloss_dict[trial] = get_mse_loss(pred, true)
                pred = np.argmax(pred, axis=-1)
                true = np.argmax(true, axis=-1) 
                pred = np.where(pred==0, -1, np.where(pred==3, 1, 0))
                true = np.where(true==0, -1, np.where(true==3, 1, 0))
                acc_list.append(get_accuracy(pred, true))
                if saved_test_results_dir_path is not None:
                    saved_path = __make_saved_path__(trial)
                    np.save(join_paths(saved_path, 'pred.npy'), pred)
                    np.save(join_paths(saved_path, 'true.npy'), true)
                    Image.fromarray(image.astype(np.uint8)).convert('RGB').save(join_paths(saved_path, 'image.png'))

    acc_list = np.array(acc_list)
    print(f'Accuracy: {acc_list.mean()}, max={acc_list.max()}, min={acc_list.min()}')
    # avg_maeloss = sum([value for value in maeloss_dict.values()])/len(maeloss_dict)
    # avg_mseloss = sum([value for value in mseloss_dict.values()])/len(mseloss_dict)
    # max_maeloss_key = max(maeloss_dict, key=maeloss_dict.get)
    # min_maeloss_key = min(maeloss_dict, key=maeloss_dict.get)
    # max_mseloss_key = max(mseloss_dict, key=mseloss_dict.get)
    # min_mseloss_key = min(mseloss_dict, key=mseloss_dict.get)
    # print(f'Average MAE loss: {avg_maeloss:.6f}, Max MAE Loss is {max_maeloss_key}: {maeloss_dict[max_maeloss_key]:.6f}, Min MAE Loss is {min_maeloss_key}: {maeloss_dict[min_maeloss_key]:.6f}')
    # print(f'Average MSE loss: {avg_mseloss:.6f}, Max MSE Loss is {max_mseloss_key}: {mseloss_dict[max_mseloss_key]:.6f}, Min MSE Loss is {min_mseloss_key}: {mseloss_dict[min_mseloss_key]:.6f}')
        
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
    if tower_name in ['image', 'i']:
        tower_name = 'i'
    elif tower_name in ['text', 't', 'caption', 'c']:
        tower_name = 'c'
    else:
        raise ValueError(f'tower_name={tower_name} is not supported.')
    

    ## Hyperparameters
    # subj id
    subj_id = configs_dict['subj_id']
    # dataset name
    dataset_name = configs_dict['dataset_name']
    # functional_space = {func1mm, func1pt8mm}
    functional_space = configs_dict['functional_space']
    assert functional_space in ['1', '1.8', 'func1mm', 'func1pt8mm'], f'functional_space should be one of [1, 1.8, func1mm, func1pt8mm], but got {functional_space}.'
    functional_space = 'func1pt8mm' if '8' in functional_space else 'func1mm'
    # embedding_space = {blip2, blipdiffusion}
    embedding_space = configs_dict['embedding_space']
    assert embedding_space in ['blip2', 'blipdiffusion', '2', 'diffusion'], f'embedding_space should be one of [blip2, blipdiffusion, 2, diffusion], but got {embedding_space}.'
    embedding_space = 'blip2' if '2' in embedding_space else 'blipdiffusion'
    # train brain decoder
    batch_size = configs_dict['train_decoder']['batch_size']
    learning_rate = configs_dict['train_decoder']['learning_rate']
    epochs = configs_dict['train_decoder']['epochs']
    # roi
    derived_type = configs_dict['NSD_ROIs']['derived_type']
    roi_name = configs_dict['NSD_ROIs']['roi_name']
    if embedding_space == 'blip2':
        thresholds = configs_dict['NSD_ROIs']['thresholds']['whole_cortex']
    elif embedding_space == 'blipdiffusion':
        if tower_name == 'c':
            thresholds = configs_dict['NSD_ROIs']['thresholds']['higher_visual_cortex']
        else:
            thresholds = configs_dict['NSD_ROIs']['thresholds']['primary_visual_cortex']
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
    rois_datapoint = fetch_nsd_rois_and_labels(functional_space=functional_space, rois_setup=rois_setup)
    mask_fMRI(functional_space=functional_space, rois_datapoint=rois_datapoint)
    
    ## Path to save
    # the path of training results
    path_info = (dataset_name, f'subj{str(subj_id).zfill(2)}', f'{derived_type}_{roi_name}', f'{rois_datapoint.labels_string}')
    saved_subj_train_result_dir_path = join_paths(train_results_dir_path, *path_info)
    formatted_date_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    saved_subj_train_result_dir_path = join_paths(saved_subj_train_result_dir_path, formatted_date_time)
    os.makedirs(saved_subj_train_result_dir_path, exist_ok=True)
    # saved_model_path = join_paths(saved_subj_train_result_dir_path, f'tower-{tower_name}_ep-{epochs}_lr-{learning_rate}.pth')
    # the path of testing results
    saved_test_results_dir_path = join_paths(test_results_dir_path, *path_info)
    os.makedirs(saved_test_results_dir_path, exist_ok=True)

    # Train-Valid and Test
    if task == 't':
        # dataloader
        train_dataloader = DataLoader(dataset=NSD_Dataset(functional_space=functional_space,
                                                          embedding_space=embedding_space,
                                                          labels_string=rois_datapoint.labels_string,
                                                          set_name='train'), 
                                      batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_dataloader  = DataLoader(dataset=NSD_Dataset(functional_space=functional_space,
                                                          embedding_space=embedding_space,
                                                          labels_string=rois_datapoint.labels_string,
                                                          set_name='test'),  
                                      batch_size=batch_size, shuffle=False, num_workers=num_workers)
        # Loss function
        # decoder_loss = Decoder_loss(w1=1, w2=1, w3=0)
        decoder_loss = torch.nn.CrossEntropyLoss() 

        # Network
        light_loader = next(iter(test_dataloader))
        image_shape = light_loader.image.shape[1:]
        masked_fmri_shape = light_loader.masked_fmri.shape[1:]
        image_embedding_shape = light_loader.image_embedding.shape[1:] 
        caption_embedding_shape = light_loader.caption_embedding.shape[1:] if embedding_space == 'blipdiffusion' else None
        print(f'image Shape  = {image_shape}')
        print(f'masked_fmri Shape  = {masked_fmri_shape}')
        print(f'image_embedding Shape = {image_embedding_shape}')
        print(f'caption_embedding Shape = {caption_embedding_shape}') if not caption_embedding_shape is None else None
        decoder_model = BraVO_Decoder(
            input_shape=masked_fmri_shape,
            image_embedding_shape=image_embedding_shape,
            caption_embedding_shape=caption_embedding_shape,
        )
        trainable_parameters = sum(p.numel() for p in decoder_model.parameters() if p.requires_grad)
        print(decoder_model)
        # decoder_model = torch.nn.DataParallel(decoder_model)
        print(f'The number of trainable parametes of {decoder_model.__class__.__name__} is {trainable_parameters}.')
        decoder_model = decoder_model.to(device=device)
        
        # Optimizer
        optimizer_of_brain_decoder = torch.optim.Adam(decoder_model.parameters(), lr=learning_rate) 
        # optimizer_of_brain_decoder = torch.optim.Adam(decoder_model.parameters(), lr=learning_rate, weight_decay=1e-3) 
        # optimizer_of_brain_decoder = torch.optim.AdamW(decoder_model.parameters(), lr=learning_rate) 
        
        # Train and Valid
        print(f'Training Brain {decoder_model.__class__.__name__} for {epochs} epochs. batch_size={batch_size}, learning_rate={learning_rate}.')
        for epoch in range(epochs):
            print(f'Tower {tower_name}, Epoch {epoch+1}/{epochs}')
            
            # # Decay Learning Rate
            # lr = learning_rate*((1-epoch/epochs)**0.9)
            # for param_group in optimizer_of_brain_decoder.param_groups:
            #     param_group['lr'] = lr
            scheduler = lr_scheduler.StepLR(optimizer_of_brain_decoder, step_size=20, gamma=0.5)
            # scheduler = lr_scheduler.LambdaLR(optimizer_of_brain_decoder, 
            #                                  lr_lambda=lambda epoch: 0.5 * (1 + torch.cos(torch.pi * torch.tensor(epoch / epochs))))
            
            # Train
            trained_model, train_loss, total_memory, mem_reserved = train(device=device, 
                                                                          model=decoder_model,
                                                                          loss_fn=decoder_loss, 
                                                                          optimizer=optimizer_of_brain_decoder, 
                                                                          dataloader=train_dataloader,
                                                                        )
            # Save the temporal trained model in each epoch
            # temporary_model_path = join_paths(saved_subj_train_result_dir_path, f'temporary_ep-{epoch+1}_lr-{learning_rate}.pth')
            # torch.save(trained_model.state_dict(), temporary_model_path)
            print(f'Train {decoder_loss.__class__.__name__} = {train_loss:.6f}')
            # Valid
            # decoder_model.load_state_dict(torch.load(temporary_model_path, weights_only=True))
            _, _ = test(device=device, 
                        # model=decoder_model, 
                        model=trained_model, 
                        dataloader=test_dataloader,
                        tower_name=tower_name, 
                        saved_test_results_dir_path=saved_test_results_dir_path
                    )
            # GPU memory usage
            print(f'GPU memory usage: {mem_reserved:.2f} GB / {total_memory:.2f} GB.')

            scheduler.step()

        # # save the finally trained model, delete the temporal trained model
        # for pth_file_path in os.listdir(saved_subj_train_result_dir_path):
        #     if pth_file_path.startswith('temporary_ep') and pth_file_path.endswith('.pth'):
        #         delete_path = join_paths(saved_subj_train_result_dir_path, pth_file_path)
        #         if os.path.exists(delete_path):
        #             os.remove(delete_path)
        # torch.save(trained_model.state_dict(), saved_model_path)

        # Test
        print(f'Testing Brain Decoder.')
        # decoder_model.load_state_dict(torch.load(saved_model_path, weights_only=True))
        total_memory, mem_reserved = test(device=device, 
                                        #   model=decoder_model, 
                                          model=trained_model, 
                                          dataloader=test_dataloader, 
                                          tower_name=tower_name,
                                          saved_test_results_dir_path=saved_test_results_dir_path
                                    )
        print(f'GPU memory usage: {mem_reserved:.2f} GB / {total_memory:.2f} GB.')
    
    # Generate
    elif task == 'g':   
        # fixed dir path
        run_files = read_json_file(run_files_path)
        fixed_dir_path = get_items_in_list_via_substrs(list(run_files.keys()), embedding_space, 'fixed')
        assert len(fixed_dir_path) == 1, f'Found multiple fixed dir path: {fixed_dir_path}.'
        fixed_dir_path = fixed_dir_path[0]
        # original strings
        test_dir_path_list = [join_paths(run_files['test'], x) for x in os.listdir(run_files['test'])]
        string_path_dict = {} # key=trial_name, value=string_path
        for test_dir_path in test_dir_path_list:
            string_path = get_items_in_list_via_substrs(os.listdir(test_dir_path), 'strings', 'json')
            assert len(string_path) == 1, f'Found multiple string path: {string_path}.'
            string_path = string_path[0]
            string_path_dict[os.path.basename(test_dir_path)] = join_paths(test_dir_path, string_path)

        # BLIP2 Generate: embedding -> caption
        if embedding_space == 'blip2':
            prompt_attention = np.load(join_paths(run_files[fixed_dir_path], 'prompt_attention.npy'), allow_pickle=True)
            prompt_embedding = np.load(join_paths(run_files[fixed_dir_path], 'prompt_embedding.npy'), allow_pickle=True)
            prompt_attentions = torch.tensor(prompt_attention, dtype=torch.float32).unsqueeze(0).to(device)
            prompt_embeddings = torch.tensor(prompt_embedding, dtype=torch.float32).unsqueeze(0).to(device)

            blip2t5_model, _, _ = load_blip_models(mode='caption') 

            for trial in tqdm(os.listdir(saved_test_results_dir_path), desc='Generating captions', leave=True):
                assert trial in string_path_dict.keys(), f'Found trial {trial} not in string_path_dict.'
                strings = read_json_file(string_path_dict[trial])
                coco_captions = strings['coco_captions']
                blip2_caption = strings['blip2_caption']
                trial_dir_path = join_paths(saved_test_results_dir_path, trial)
                
                # image_embedding = np.load(join_paths(trial_dir_path, 'pred.npy'), allow_pickle=True)
                ### test
                image_embedding = np.load(join_paths(trial_dir_path, 'true.npy'), allow_pickle=True)
                ### test
                image_embeds = torch.tensor(image_embedding, dtype=torch.float32).unsqueeze(0).to(device)
                output_text = blip2t5_model.generate_captions_via_embedding(
                    image_embeds=image_embeds,
                    prompt_embeddings=prompt_embeddings,
                    prompt_attentions=prompt_attentions,
                    max_length=configs_dict['blip2']['max_length'], 
                    min_length=configs_dict['blip2']['min_length']
                )[0]
                print()
                print('coco  : ', coco_captions)
                print('blip2 :', blip2_caption)
                print('output: ', output_text)

            del blip2t5_model

        # BLIP Diffusion Generate: embedding -> image
        elif embedding_space == 'blipdiffusion':
            pass
        # def __merge_images_with_separators__(
        #         images_dict : dict[str, Image.Image], saved_dir_path : str,
        #         separator_width : int = 10, separator_color : tuple[int, int, int] = (255, 255, 255)
        # ) -> None:
        #     names, images = [], []
        #     for name, image in images_dict.items():
        #         names.append(name)
        #         images.append(image)
        #     name = '__'.join(names)
        #     total_width = sum(image.width for image in images) + separator_width * (len(images) - 1)  
        #     max_height = max(image.height for image in images)  
        #     new_img = Image.new('RGB', (total_width, max_height), separator_color)  
        #     current_x = 0  
        #     for img in images:  
        #         new_img.paste(img, (current_x, 0))  
        #         current_x += img.width + separator_width  
        #     new_img.save(join_paths(saved_dir_path, f'{name}.png'))
    
        # blip_diffusion_model, _, _ = load_blip_models(mode='diffusion')
        # blip2t5_model, blip2t5_vis_processors, _ = load_blip_models(mode='caption') 
        # blip2itm_model, blip2itm_vis_processors, blip2itm_text_processors = load_blip_models(mode='matching')
    
        # test_dirs_path_dict = {int(path):join_paths(saved_test_results_dir_path, path) for path in os.listdir(saved_test_results_dir_path) if os.path.isdir(join_paths(saved_test_results_dir_path, path))}
        # sorted_keys = sorted(test_dirs_path_dict.keys())
        # test_dirs_path_dict = {key:test_dirs_path_dict[key] for key in sorted_keys}
        # for index, dir_path in test_dirs_path_dict.items():
        #     if index in [12,16,30,52,85,151,222,531]:
        #         print(f'Generating {index} / {len(test_dirs_path_dict)}')
        #         coco_matrix = np.load(join_paths(dir_path, 'coco.npy'), allow_pickle=True)
        #         coco = Image.fromarray(coco_matrix).convert('RGB')
        #         bravo_img = np.load(join_paths(dir_path, 'bravo_img.npy'), allow_pickle=True)
        #         bravo_cap = np.load(join_paths(dir_path, 'bravo_cap.npy'), allow_pickle=True)
        #         blip_img  = np.load(join_paths(dir_path, 'blip_img.npy' ), allow_pickle=True)
        #         blip_cap  = np.load(join_paths(dir_path, 'blip_cap.npy' ), allow_pickle=True)

        #         hidden_state_dict = {
        #             'blipI+blipC'   : BLIP_Prior_Tools.concatenate_embeddings(img_emb=blip_img , txt_emb=blip_cap),
        #             'blipI+bravoC'  : BLIP_Prior_Tools.concatenate_embeddings(img_emb=blip_img , txt_emb=bravo_cap),
        #             'bravoI+blipC'  : BLIP_Prior_Tools.concatenate_embeddings(img_emb=bravo_img, txt_emb=blip_cap ),
        #             'nullI+nullC'   : BLIP_Prior_Tools.concatenate_embeddings(img_emb=null_img_embedding, txt_emb=null_cap_embedding),
        #             'nullI+blipC'   : BLIP_Prior_Tools.concatenate_embeddings(img_emb=null_img_embedding, txt_emb=blip_cap),
        #             'blipI+nullC'   : BLIP_Prior_Tools.concatenate_embeddings(img_emb=blip_img, txt_emb=null_cap_embedding),
        #             'nullI+bravoC'  : BLIP_Prior_Tools.concatenate_embeddings(img_emb=null_img_embedding, txt_emb=bravo_cap),
        #             'bravoI+nullC'  : BLIP_Prior_Tools.concatenate_embeddings(img_emb=bravo_img, txt_emb=null_cap_embedding),
        #             'bravoI+bravoC' : BLIP_Prior_Tools.concatenate_embeddings(img_emb=bravo_img, txt_emb=bravo_cap),
        #         }
        #         images_dict = {
        #             'coco' : coco
        #         }
        #         captions_dict = {}
        #         for key in hidden_state_dict:
        #             hidden_state = hidden_state_dict[key]
        #             assert hidden_state.shape == position_embeddings.shape, f'hidden_state.shape={hidden_state.shape} != position_embeddings.shape={position_embeddings.shape}'
        #             # TODO null不用加position_embeddings
        #             hidden_state += position_embeddings
        #             hidden_state = torch.from_numpy(hidden_state).unsqueeze(0).to(device)
        #             assert hidden_state.shape == uncond_embedding.shape, f'{hidden_state.shape} != {uncond_embedding.shape}'
        #             generated_image = blip_diffusion_model.generate_image_via_embedding(
        #                                                     uncond_embedding=uncond_embedding,
        #                                                     hidden_states=hidden_state,
        #                                                     causal_attention_mask=causal_attention_mask,
        #                                                     seed=iter_seed,
        #                                                     guidance_scale=guidance_scale,
        #                                                     height=coco_matrix.shape[0],
        #                                                     width=coco_matrix.shape[1],
        #                                                     num_inference_steps=num_inference_steps,
        #                                                 )
        #             images_dict[key] = generated_image.convert('RGB')
        #             start_time = time.time()
        #             image = blip2t5_vis_processors['eval'](images_dict[key]).unsqueeze(0).to(device)
        #             caption = blip2t5_model.generate({'image' : image, 'prompt' : prompt},
        #                                                   max_length=100, min_length=30)
        #             captions_dict[key] = caption[0]
        #             end_time = time.time()
        #             print(f'Time taken to generate caption for {key}: {end_time - start_time:.4f} seconds.')

        #         __merge_images_with_separators__(images_dict=images_dict, saved_dir_path=dir_path)
        #         captions_json_path = join_paths(dir_path, 'captions_and_itmscores.json')
        #         blip_caption_dict = {'blip_caption' : read_json_file(captions_json_path)['blip_caption']}
        #         all_captions = merge_dicts_if_no_conflict(dict1=blip_caption_dict, dict2=captions_dict)
               
        #         # image-text matching score                
        #         coco = blip2itm_vis_processors['eval'](coco).unsqueeze(0).to(device)
        #         captions_itmscores_dict = {}
        #         for tag, text in all_captions.items():
        #             print(tag)
        #             print(f"\t{text}")
        #             captions_itmscores_dict[tag] = {'caption' : text, 'itm_score' : None}
        #             text = blip2itm_text_processors['eval'](text)
        #             itm_output = blip2itm_model({'image': coco, 'text_input': text}, match_head='itm')
        #             itm_score = torch.nn.functional.softmax(itm_output, dim=1)[:, 1].item()
        #             captions_itmscores_dict[tag]['itm_score'] = itm_score
        #             print(f"\tITM Score: {itm_score}")
        #         write_json_file(captions_json_path, captions_itmscores_dict)
        #         print()
    else:
        raise ValueError(f'Task should be either [train test generate generation], but got {task}.')
    
if __name__ == '__main__':
    main()
    print('Done.\n\n')
