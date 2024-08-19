import os
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from losses import Decoder_loss
from config import configs_dict
from models import load_blip_models
from dataset import make_paths_dict, fetch_roi_files_and_labels, NSD_Dataset
from models import device, get_GPU_memory_usage, BraVO_Decoder
from utils import train_results_dir_path, test_results_dir_path
from utils import join_paths, check_and_make_dirs, write_json_file, read_json_file
from analyze import analyze_embeddingPriori_and_maskedBrain, analyze_all_blipdiffusion_embeddings, analyze_change_embedding


def train(
    device : torch.device,
    model : torch.nn.Module,
    loss_fn : torch.nn.modules.loss._Loss,
    optimizer : torch.optim.Optimizer,
    train_dataloader : DataLoader,
    maskeddata_mean : float,
    maskeddata_std  : float,
    embeddings_mean : float,
    embeddings_std  : float
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
        masked_data = (masked_data - maskeddata_mean) / maskeddata_std
        # Forward
        pred_embedding, mean, log_var  = model(masked_data)
        multimodal_embedding = multimodal_embedding.view(multimodal_embedding.shape[0], -1)
        multimodal_embedding = (multimodal_embedding - embeddings_mean) / embeddings_std
        # Compute loss
        # 如果是图像的话，还有风格损失 感知损失 结构损失等
        loss = loss_fn(input=pred_embedding, target=multimodal_embedding, mean=mean, log_var=log_var)
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
    test_dataloader : DataLoader,
    saved_test_results_dir_path : str,
    maskeddata_mean : float,
    maskeddata_std  : float,
    embeddings_mean : float,
    embeddings_std  : float
) -> tuple[float, float, float]:
    """
    """
    model.eval()
    metrics_dict = {} # {key=index, value=Euclidean distance}
    mem_reserved_list = []
    with torch.no_grad():
        for index, masked_data, image_data, canny_data, multimodal_embedding in tqdm(test_dataloader, desc='Testing', leave=True):
            # Load data to device and set the dtype as float32
            tensors = [masked_data, image_data, canny_data, multimodal_embedding]
            tensors = list(map(lambda t: t.to(device=device, dtype=torch.float32), tensors))
            masked_data, image_data, canny_data, multimodal_embedding = tensors
            # Scale the masked_data
            masked_data = (masked_data - masked_data.mean()) / masked_data.std()
            # Forward
            pred_embedding, mean, log_var  = model(masked_data)
            # pred_image = (pred_image * 255).round()
            # # Map pred_embedding into original space
            # pred_embedding = Bijection_ND_CDF(backward_mean=embeddings_mean, 
            #                                   backward_std=embeddings_std).eval().backward(pred_embedding)
            pred_embedding = pred_embedding.view(pred_embedding.shape[0], multimodal_embedding.shape[1], multimodal_embedding.shape[2])
            # for idx, mse in zip(index, mse_loss):
                # metrics_dict[idx.item()] = mse.item()
            # save the results
            index = index.cpu().numpy()
            pred_embedding = pred_embedding.cpu().numpy()
            multimodal_embedding = multimodal_embedding.cpu().numpy()
            for idx, pred, true in zip(index, pred_embedding, multimodal_embedding):
                metrics_dict[idx] = np.mean(np.power(pred - true, 2))
                saved_path = join_paths(saved_test_results_dir_path, str(idx))
                check_and_make_dirs(saved_path)
                np.save(join_paths(saved_path, 'pred.npy'), pred)
                np.save(join_paths(saved_path, 'true.npy'), true)
                # pred = Image.fromarray((pred * 255).astype(np.uint8))
                # image = Image.fromarray((image * 255).astype(np.uint8))
                # pred.save(join_paths(saved_path, f'pred.png'))
                # image.save(join_paths(saved_path, f'image.png'))

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
    batch_size = configs_dict['train_decoder']['batch_size']
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
    
    ## Data
    train_trial_path_dict, test_trial_path_dict, rois_path_dict, uncond_embedding = make_paths_dict(subj_id=subj_id, dataset_name=dataset_name)
    mask_path_list, labels_string = fetch_roi_files_and_labels(derived_type=derived_type, roi_name=roi_name, thresholds=thresholds, rois_path_dict=rois_path_dict)
    train_dataloader = DataLoader(dataset=NSD_Dataset(train_trial_path_dict, mask_path_list, thresholds), 
                                batch_size=batch_size, shuffle=False, num_workers=6)
    test_dataloader = DataLoader(dataset=NSD_Dataset(test_trial_path_dict, mask_path_list, thresholds), 
                                batch_size=batch_size, shuffle=False, num_workers=6)

    ## Path to save
    saved_subj_train_result_dir_path = join_paths(train_results_dir_path, dataset_name, f'subj{str(subj_id).zfill(2)}', f'{derived_type}_{roi_name}', f'{labels_string}')
    check_and_make_dirs(saved_subj_train_result_dir_path)
    # the path of saving the trained model
    saved_model_path = join_paths(saved_subj_train_result_dir_path, f'ep-{epochs}_bs-{batch_size}_lr-{learning_rate}.pth')
    # the path of saving the priori
    saved_priori_path = join_paths(saved_subj_train_result_dir_path, r'priori.json')
    # path to save the prediected fMRI(whole brain)
    saved_test_results_dir_path = join_paths(test_results_dir_path, dataset_name, f'subj{str(subj_id).zfill(2)}', f'{derived_type}_{roi_name}', f'{labels_string}')
    check_and_make_dirs(saved_test_results_dir_path)

    ## Algorithm
    if task in ['train', 'test']: # to save memory usage
        # Priori from masked fMRI and BLIP diffusion's embedding on COCO
        if not os.path.exists(saved_priori_path):
            priori = analyze_embeddingPriori_and_maskedBrain(dataloader=train_dataloader, save_figs_path=saved_subj_train_result_dir_path)
            write_json_file(path=saved_priori_path, data=priori)
        else:
            priori = read_json_file(path=saved_priori_path)
        maskeddata_mean = priori['masked_data']['mean']
        maskeddata_std = priori['masked_data']['std']
        embeddings_mean = priori['multimodal_embedding']['mean']
        embeddings_std = priori['multimodal_embedding']['std']
        # Network
        light_loader = next(iter(test_dataloader))
        input_shape  = light_loader[1].shape[1:]  # The shape of masked_data
        # output_shape = light_loader[2].shape[1:] # The shape of image_data
        output_shape = light_loader[-1].shape[1:] # The shape of multimodal_embedding
        bravo_decoder_model = BraVO_Decoder(input_shape=input_shape, output_shape=output_shape)
        print(bravo_decoder_model)
        trainable_parameters = sum(p.numel() for p in bravo_decoder_model.parameters() if p.requires_grad)
        bravo_decoder_model = bravo_decoder_model.to(device=device)
        print(f'The number of trainable parametes is {trainable_parameters}.')
        # Loss function
        decoder_loss = Decoder_loss(w1=1, w2=1)
        # Optimizer
        optimizer_of_brain_decoder = torch.optim.Adam(bravo_decoder_model.parameters(), lr=learning_rate) 
    else:
        pass

    # Train
    if task == 'train':
        print(f'Training Brain Decoder for {epochs} epochs.')
        for epoch in range(epochs):
            start_time = time.time()
            lr = learning_rate*((1-epoch/epochs)**0.9)
            for param_group in optimizer_of_brain_decoder.param_groups:
                param_group['lr'] = lr
            trained_model, train_loss, total_memory, mem_reserved = train(device=device, 
                                                                          model=bravo_decoder_model, 
                                                                          loss_fn=decoder_loss, 
                                                                          optimizer=optimizer_of_brain_decoder, 
                                                                        #   train_dataloader=train_dataloader,
                                                                          ### test
                                                                          train_dataloader=test_dataloader,
                                                                          ### test
                                                                          maskeddata_mean=maskeddata_mean,
                                                                          maskeddata_std=maskeddata_std,
                                                                          embeddings_mean=embeddings_mean,
                                                                          embeddings_std=embeddings_std
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
    # Test
    elif task == 'test':
        print(f'Testing Brain Decoder.')
        # load the trained model
        ### test
        # saved_model_path = join_paths(saved_subj_train_result_dir_path, 'temporary_ep-25_lr-0.0001.pth')
        ### test
        bravo_decoder_model.load_state_dict(torch.load(saved_model_path))
        Euclidean_distance, total_memory, mem_reserved = test(device=device, 
                                          model=bravo_decoder_model, 
                                          test_dataloader=test_dataloader, 
                                          saved_test_results_dir_path=saved_test_results_dir_path,
                                          maskeddata_mean=maskeddata_mean,
                                          maskeddata_std=maskeddata_std,
                                          embeddings_mean=embeddings_mean,
                                          embeddings_std=embeddings_std
                                    )
        print(f'Averaged Euclidean distance = {Euclidean_distance:.4f}.')
        print(f'GPU memory usage: {mem_reserved:.2f} GB / {total_memory:.2f} GB.')
    # Generate
    elif task == 'generate' or task == 'generation':
        blip_diffusion_model, _, _ = load_blip_models(mode = 'diffusion')
        for index, files_path in test_trial_path_dict.items():
            print(f'Generating {index+1} / {len(test_trial_path_dict)}.')
            image_path = files_path['image']
            canny_path = files_path['canny']
            strings_path = files_path['strings']
            pred_multimodal_embedding_path = join_paths(saved_test_results_dir_path, str(index), 'pred.npy')
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
            generated_image[0].save(join_paths(saved_test_results_dir_path, str(index), 'output.png'))
    # Analyze
    elif task == 'analyze':
        analyze_all_blipdiffusion_embeddings(train_dataloader=train_dataloader, test_dataloader=test_dataloader)
        analyze_change_embedding(dataloader=test_dataloader, uncond_embedding=uncond_embedding)
    else:
        raise ValueError(f'Task should be either train or test, but got {task}.')
    
if __name__ == '__main__':
    main()
   