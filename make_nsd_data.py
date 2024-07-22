import os
import h5py
import time
import torch
import shutil
import scipy.io
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader

from dataset import BLIP2_Dataset
from models import device, load_blip_models
from utils import NSD_dir_path, BraVO_saved_dir_path
from utils import join_paths, read_nii_file, save_nii_file, check_and_make_dirs, read_json_file, write_json_file, merge_dicts_if_no_conflict, get_items_in_list_via_substrs

# Load blip2 model
BLIP2_Encoder_model, vis_processors, txt_processors = load_blip_models(mode = 'encoder')

class NSD_DATA():
    def __init__(self, NSD_dir_path : str = NSD_dir_path, subj_id : int | str = None) -> None:
        super().__init__()
        # subj_id
        self.subj = ''.join(['subj0', str(subj_id)])
        
        ## nsddata
        self.nsddata_dir_path = join_paths(NSD_dir_path, 'nsddata')
        self.nsddata_ppdata_dir_path = join_paths(self.nsddata_dir_path, 'ppdata')
        self.nsddata_freesurfer_dir_path = join_paths(self.nsddata_dir_path, 'freesurfer')
        # Info: https://cvnlab.slite.page/p/fRv4lz5V2F/Behavioral-data#2bdd55ef
        self.behav_responses_tsv_file_path = join_paths(self.nsddata_ppdata_dir_path, self.subj, 'behav', 'responses.tsv')
        # https://cvnlab.slite.page/p/X_7BBMgghj/ROIs#c5518e3e
        self.roi_files_path = join_paths(self.nsddata_ppdata_dir_path, self.subj, 'func1mm', 'roi')
        # https://cvnlab.slite.page/p/X_7BBMgghj/ROIs#2da19afb
        self.labels_path = join_paths(self.nsddata_freesurfer_dir_path, self.subj, 'label')
        # https://cvnlab.slite.page/p/X_7BBMgghj/ROIs#65b75445
        self.templates_path = join_paths(self.nsddata_dir_path, 'templates')
        # Info: https://cvnlab.slite.page/p/NKalgWd__F/Experiments#b0ea56ab
        self.expdesign_mat_file_path = join_paths(self.nsddata_dir_path, 'experiments', 'nsd', 'nsd_expdesign.mat')
        # Info: https://cvnlab.slite.page/p/NKalgWd__F/Experiments#bf18f984
        self.stim_info_csv_file_path = join_paths(self.nsddata_dir_path, 'experiments', 'nsd', 'nsd_stim_info_merged.csv')

        ## nsddata_betas
        # Info: https://cvnlab.slite.page/p/6CusMRYfk0/Functional-data-NSD#035bbb1e
        self.nsddata_betas_ppdata_dir_path = join_paths(NSD_dir_path, 'nsddata_betas', 'ppdata', self.subj)
        
        ## nsddata_stimuli
        # Info: https://cvnlab.slite.page/p/NKalgWd__F/Experiments#b44e32c0
        self.nsddata_stimuli_hdf5_file_path = join_paths(NSD_dir_path, 'nsddata_stimuli', 'stimuli', 'nsd', 'nsd_stimuli.hdf5')
       
        ## COCO annotation
        # Ref: https://cvnlab.slite.page/p/M3ZvPmfgU3/General-Information#1d5942f6
        self.coco_annotation_dir_path = join_paths(NSD_dir_path, 'nsddata_stimuli', 'stimuli', 'nsd', 'annotations')

        ## saved path of this subject
        self.subject_saved_dir_path = join_paths(BraVO_saved_dir_path, self.subj+'_pairs')
        check_and_make_dirs(self.subject_saved_dir_path)

        # make fMRI-image-caption pairs
        self.make_pairs()

    def read_behav_responses_tsv(self) -> pd.core.frame.DataFrame:
        start_time = time.time()
        data_frame = pd.read_csv(self.behav_responses_tsv_file_path, sep='\t', encoding='utf-8')
        subj_numpyINT64 = np.int64(self.subj[-1])
        assert (data_frame['SUBJECT'] == subj_numpyINT64).all(), print(f'Subject id in tsv file is not correct.') # subj 1~8
        data_frame.drop(columns=['SUBJECT'], inplace=True)
        # Some columns are not needed
        data_frame.drop(columns=['TIME'], inplace=True)
        data_frame.drop(columns=['MEMORYRECENT'], inplace=True)
        data_frame.drop(columns=['MEMORYFIRST'], inplace=True)
        data_frame.drop(columns=['TOTAL1'], inplace=True)
        data_frame.drop(columns=['TOTAL2'], inplace=True)
        data_frame.drop(columns=['BUTTON'], inplace=True)
        data_frame.drop(columns=['MISSINGDATA'], inplace=True)
        end_time = time.time()
        print(f'It took {end_time - start_time:.2f} seconds to read {self.behav_responses_tsv_file_path}.')
        return data_frame

    def read_expdesign_mat(self) -> dict[str, any]:
        start_time = time.time()
        mat_contents = scipy.io.loadmat(self.expdesign_mat_file_path)
        end_time = time.time()
        print(f'It took {end_time - start_time:.2f} seconds to read {self.expdesign_mat_file_path}.')
        return mat_contents
    
    def read_stim_info_csv(self) -> dict[int, int]:
        start_time = time.time()
        data_frame = pd.read_csv(self.stim_info_csv_file_path)
        # cocoId: is the ID number assigned to this image in the COCO database.
        # nsdId: is the 0-based index of the image into the full set of 73k images used in the NSD experiment. Values are the same as column 1. (Note that in some other cases, 73k IDs are specified as 1-based. Here the IDs are specified as 0-based.)
        end_time = time.time()
        print(f'It took {end_time - start_time:.2f} seconds to read {self.stim_info_csv_file_path}.')
        return dict(zip(data_frame['nsdId'], data_frame['cocoId']))

    def read_betas(self, session_id : int, space_type : str = 'func1mm') -> tuple[str, np.ndarray]:
        start_time = time.time()
        if space_type == 'func1mm':
            # func1mm and func1pt8mm: https://cvnlab.slite.page/p/6CusMRYfk0/Functional-data-NSD#9065a649
            func1mm_dir_path = join_paths(self.nsddata_betas_ppdata_dir_path, 'func1mm')
            betas_fithrf_GLMdenoise_RR_dir_path = join_paths(func1mm_dir_path, 'betas_fithrf_GLMdenoise_RR')
            # Info: https://cvnlab.slite.page/p/6CusMRYfk0/Functional-data-NSD#3e1740b1
            file_name = f'betas_session{str(session_id).zfill(2)}.nii.gz'
            file_path = join_paths(betas_fithrf_GLMdenoise_RR_dir_path, file_name)
            header, data = read_nii_file(file_path) # dims=(145, 186, 148, 750); dtype=float64
            assert np.iinfo(np.int16).min <= np.min(data) and np.iinfo(np.int16).max >= np.max(data), print(f'Data range is not within int16 range.')
            data = data.astype(np.int16)
            data = np.transpose(data, (3, 0, 1, 2)) # dims=(750, 145, 186, 148)
        else:
            raise NotImplementedError(f'Space type: {space_type} is not supported.')
        end_time = time.time()
        print(f'It took {end_time - start_time:.2f} seconds to read {file_path}.')
        return space_type, data
        
    def read_stimuli_hdf5(self) -> np.ndarray:
        start_time = time.time()
        with h5py.File(self.nsddata_stimuli_hdf5_file_path, 'r') as f:
            # imgBrick is 3 channels x 425 pixels x 425 pixels x 73,000 images and is in uint8 format. 
            # These images are shown on a gray background with RGB value (127,127,127).
            imgBrick = f['imgBrick'][:]
        end_time = time.time()
        print(f'It took {end_time - start_time:.2f} seconds to read {self.nsddata_stimuli_hdf5_file_path}.')
        return imgBrick
    
    def read_coco_annotation(self) -> tuple[dict[int, list[str]], dict[int, list[str]]]:
        def __extract_captions__(captions_annotations : list[dict[str, any]]) -> dict[int, list[str]]:
            annotations = {} # {key=id : value=[caption1, caption2, ...]}
            # some pictures have multiple captions
            for ca in captions_annotations:
                if not ca['image_id'] in annotations:
                    annotations[ca['image_id']] = [ca['caption']] 
                else:
                    annotations[ca['image_id']].append(ca['caption'])
            return annotations
        
        def __extract_categories__(annotations_list : list[dict[str, any]], categories_list : list[dict[str, any]]) -> dict[int, list[dict[str, any]]]:
            categories_dict = {}
            for categories in categories_list:
                categories_dict[categories['id']] = {'supercategory':categories['supercategory'], 'name':categories['name']}
            instances_category = {}
            for annotation in annotations_list:
                category_id = annotation['category_id']
                value = {'supercategory':categories_dict[category_id]['supercategory'], 'name':categories_dict[category_id]['name'], 'area':annotation['area']}
                if not annotation['image_id'] in instances_category:
                    instances_category[annotation['image_id']] = [value]
                else:
                    instances_category[annotation['image_id']].append(value)
            return instances_category

        start_time = time.time()
        # captions
        captions_train2017 = read_json_file(path=join_paths(self.coco_annotation_dir_path, 'captions_train2017.json'))
        captions_val2017 = read_json_file(path=join_paths(self.coco_annotation_dir_path, 'captions_val2017.json'))
        captions_train_annotations = captions_train2017['annotations']
        captions_val_annotations = captions_val2017['annotations']
        train_annotations = __extract_captions__(captions_train_annotations)
        val_annotations = __extract_captions__(captions_val_annotations)
        # captions_dict is {key=id : value=[caption1, caption2, ...]}
        captions_dict = merge_dicts_if_no_conflict(train_annotations, val_annotations) 

        # categories
        instances_train2017 = read_json_file(path=join_paths(self.coco_annotation_dir_path, 'instances_train2017.json'))
        instances_val2017 = read_json_file(path=join_paths(self.coco_annotation_dir_path, 'instances_val2017.json'))
        annotations_train2017 = instances_train2017['annotations']
        annotations_val2017 = instances_val2017['annotations']
        categories_train2017 = instances_train2017['categories']
        categories_val2017 = instances_val2017['categories']
        train_categories = __extract_categories__(annotations_train2017, categories_train2017)
        val_categories = __extract_categories__(annotations_val2017, categories_val2017)
        # categories_dict is {key=id : value={supercategory, name}}
        categories_dict = merge_dicts_if_no_conflict(train_categories, val_categories)
        
        end_time = time.time()
        print(f'It took {end_time - start_time:.2f} seconds to read {self.coco_annotation_dir_path}.')
        return captions_dict, categories_dict
    
    def read_ROIs(self) -> None:
        start_time = time.time()
        # saved path for ROIs
        saved_rois_path = join_paths(self.subject_saved_dir_path, 'ROIs')
        check_and_make_dirs(saved_rois_path)

        # https://cvnlab.slite.page/p/X_7BBMgghj/ROIs#6824f30b and https://cvnlab.slite.page/p/X_7BBMgghj/ROIs#73ec0b9c
        rois_path_list = [join_paths(self.roi_files_path, x) for x in os.listdir(self.roi_files_path) if x.endswith('.nii.gz')]
        # https://cvnlab.slite.page/p/X_7BBMgghj/ROIs#929e891c and https://cvnlab.slite.page/p/X_7BBMgghj/ROIs#2da19afb
        labels_path_list = [join_paths(self.labels_path, x) for x in os.listdir(self.labels_path)]
        # https://cvnlab.slite.page/p/X_7BBMgghj/ROIs#65b75445
        templates_path_list = [join_paths(self.templates_path, x) for x in os.listdir(self.templates_path)]

        def __get_rois__(tag : str, ROIs_type : str) -> None:
            # saved path 
            ROIs_type = ROIs_type.lower()
            saved_path = join_paths(saved_rois_path, ROIs_type, tag)
            check_and_make_dirs(saved_path)

            # Surface/Volume ROIs
            roi_path_list = get_items_in_list_via_substrs(rois_path_list, tag) # itself, lh, rh
            assert len(roi_path_list) == 3, print(f'There are {len(roi_path_list)} ROIs for {tag}.')
            for roi_path in roi_path_list:
                shutil.copy(roi_path, saved_path)

            # Surface labels
            if ROIs_type.lower() == 'surface':
                label_path_list = get_items_in_list_via_substrs(labels_path_list, tag, 'ctab')
            # Volume labels
            elif ROIs_type.lower() == 'volume':
                label_path_list = get_items_in_list_via_substrs(templates_path_list, tag, 'ctab')
            else:
                raise NotImplementedError(f'ROIs \' type: {ROIs_type} is not supported.')
            assert len(label_path_list) == 1, print(f'There should be only one label file for {tag}, the label_path_list is {label_path_list}.')
            label_path = label_path_list[0]
            label_tags_dict = {-1 : 'non-cortical voxels'} # {key=label_id : value=name}
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.replace('\n', '').split(' ')
                    line = [s.replace('\t', '') for s in line if s]
                    assert len(line) == 2, print(f'Invalid line: {line} of path = {label_path}')
                    label_tags_dict[int(line[0])] = line[-1]
            write_json_file(path = join_paths(saved_path, 'label_tags.json'), data = label_tags_dict)

        ## Surface-derived ROIs
        # https://cvnlab.slite.page/p/X_7BBMgghj/ROIs#b7d9d230
        # corticalsulc is a folding-based atlas defined based on the curvature of fsaverage (sulci, gyri). It labels major sulci and some gyri throughout the whole cortex.
        __get_rois__(tag='corticalsulc', ROIs_type='surface')

        # https://cvnlab.slite.page/p/X_7BBMgghj/ROIs#bfdf19b3
        # https://cvnlab.slite.page/p/X_7BBMgghj/ROIs#abe63daa
        # floc-bodies is a collection of manually drawn ROIs based on results of the floc experiment. These ROIs consist of EBA, FBA-1, FBA-2, and mTL-bodies ("mid temporal lobe bodies"). These ROIs were the result of (liberal) thresholding at t > 0 (flocbodiestval).
        __get_rois__(tag='floc-bodies', ROIs_type='surface')
        
        # https://cvnlab.slite.page/p/X_7BBMgghj/ROIs#0ce85065
        # https://cvnlab.slite.page/p/X_7BBMgghj/ROIs#effe6170
        # floc-faces is a collection of manually drawn ROIs based on results of the floc experiment. These ROIs consist of OFA, FFA-1, FFA-2, mTL-faces ("mid temporal lobe faces"), and aTL-faces ("anterior temporal lobe faces"). These ROIs were the result of (liberal) thresholding at t > 0 (flocfacestval).
        __get_rois__(tag='floc-faces', ROIs_type='surface')
        
        # https://cvnlab.slite.page/p/X_7BBMgghj/ROIs#02d28f14
        # https://cvnlab.slite.page/p/X_7BBMgghj/ROIs#97aab6fe
        # floc-places is a collection of manually drawn ROIs based on results of the floc experiment. These ROIs consist of OPA, PPA, and RSC. These ROIs were the result of (liberal) thresholding at t > 0 (flocplacestval).
        __get_rois__(tag='floc-places', ROIs_type='surface')

        # https://cvnlab.slite.page/p/X_7BBMgghj/ROIs#88af4df2
        # https://cvnlab.slite.page/p/X_7BBMgghj/ROIs#d2ff50ea
        # floc-words is a collection of manually drawn ROIs based on results of the floc experiment. These ROIs consist of OWFA, VWFA-1, VWFA-2, mfs-words ("mid fusiform sulcus words"), and mTL-words ("mid temporal lobe words"). These ROIs were the result of (liberal) thresholding at t > 0 (flocwordtval).
        __get_rois__(tag='floc-words', ROIs_type='surface')

        # https://cvnlab.slite.page/p/X_7BBMgghj/ROIs#f3040279
        # HCP_MMP1 is the Glasser et al., Nature, 2016 atlas.
        __get_rois__(tag='HCP_MMP1', ROIs_type='surface')

        # https://cvnlab.slite.page/p/X_7BBMgghj/ROIs#63eb20bf
        # Kastner2015 is the Wang et al., Cerebral Cortex, 2015 atlas.
        __get_rois__(tag='Kastner2015', ROIs_type='surface')

        # https://cvnlab.slite.page/p/X_7BBMgghj/ROIs#5641b201
        # nsdgeneral is a general ROI that was manually drawn on fsaverage covering voxels responsive to the NSD experiment in the posterior aspect of cortex.
        __get_rois__(tag='nsdgeneral', ROIs_type='surface')

        # https://cvnlab.slite.page/p/X_7BBMgghj/ROIs#eac783df
        # https://cvnlab.slite.page/p/X_7BBMgghj/ROIs#866c3039
        # prf-eccrois is a collection of manually drawn ROIs that cover the exact same cortical extent as the prf-visualrois ROIs. These ROIs consist of ecc0pt5, ecc1, ecc2, ecc4, and ecc4+, and indicate increasing “concentric” ROIs that cover up to 0.5°, 1°, 2°, 4°, and >4° eccentricity.
        __get_rois__(tag='prf-eccrois', ROIs_type='surface')

        # https://cvnlab.slite.page/p/X_7BBMgghj/ROIs#208cb65c
        # https://cvnlab.slite.page/p/X_7BBMgghj/ROIs#e34545b9
        # prf-visualrois is a collection of manually drawn ROIs based on results of the prf experiment. These ROIs consist of V1v, V1d, V2v, V2d, V3v, V3d, and hV4. These ROIs extend from the fovea (0° eccentricity) to peripheral cortical regions that still exhibit sensible signals in the prf experiment given the limited stimulus size (this means up to about ~5-6° eccentricity).
        __get_rois__(tag='prf-visualrois', ROIs_type='surface')

        # https://cvnlab.slite.page/p/X_7BBMgghj/ROIs#9b363291
        # streams is an anatomical atlas that labels various “streams” in visual cortex. It is largely based on fsaverage folding but also takes into account the b3 noise ceiling results to ensure that the regions generally cover where there are stimulus-related signals. More details are provided below.
        __get_rois__(tag='streams', ROIs_type='surface')

        ## Volume-derived ROIs
        # https://cvnlab.slite.page/p/X_7BBMgghj/ROIs#928e15e1 
        # MTL provides manual segmentation of various regions in the medial temporal lobe, including hippocampal subfields. A expert human annotator used the raw high-resolution T2 volumes and manually segmented regions according to Berron et al., NeuroImage Clinical, 2017 for each of the 8 NSD subjects. These ROI labelings were then co-registered to the official isotropic T2 volume space and processed.
        __get_rois__('MTL', ROIs_type='volume')

        # https://cvnlab.slite.page/p/X_7BBMgghj/ROIs#1bbdd8c4
        # thalamus provides manual segmentation of thalamic regions: LGN, SC, and pulvinar (several subdivisions). Regions were defined in each hemisphere by an expert. Definition was based mostly on T1 anatomical data, but for the pulvinar, MNI-based results from other datasets were projected to each subject to aid ROI definition. Note that as a matter of definition, the ventral pulvinar is most correlated with early visual cortex; the dorsal lateral pulvinar is most correlated with the attention network; and the dorsal medial pulvinar is most correlated with the default-mode network. Additional information: LGN and SC were defined based on T1 and T2 image contrast. For the ventral pulvinar, the extent of the pulvinar was defined based on T1 and T2 contrast and then constrained to the ventral lateral portion based on the extent of the two ventral pulvinar maps reported in Arcaro et al., Journal of Neuroscience, 2015. The dorsolateral pulvinar was based on the average correlation with IPS maps; and the dorsomedial pulvinar was based on average correlation with precuneus (as reported in Arcaro et al. Nature Communications 2018).
        __get_rois__('thalamus', ROIs_type='volume')

        end_time = time.time()
        print(f'It took {end_time - start_time:.2f} seconds to get ROIs to {saved_rois_path}.')

    def make_pairs(self) -> None:
        """
        fMRI <--> image <--> text
        """        
        ## behav_responses_tsv
        responses = self.read_behav_responses_tsv()
        first_row = responses.iloc[0]
        # Info: https://cvnlab.slite.page/p/fRv4lz5V2F/Behavioral-data#cda8328e
        column_of_RUN = first_row.index.get_loc('RUN')    
        column_of_TRIAL = first_row.index.get_loc('TRIAL')  
        column_of_73KID = first_row.index.get_loc('73KID') # the 73k IDs are provided as 1-based indices
        column_of_ISOLD = first_row.index.get_loc('ISOLD') # 0 was novel, 1 was old.
        column_of_ISCORRECT = first_row.index.get_loc('ISCORRECT') # 0 was incorrect, 1 was correct.

        ## expdesign_mat
        expdesign = self.read_expdesign_mat()
        # Info: https://cvnlab.slite.page/p/NKalgWd__F/Experiments#f06eb84b
        sharedixs = np.squeeze(expdesign['sharedix']) - 1 # 0-based index

        ## stim_info_csv
        stim_info = self.read_stim_info_csv() # {key=nsdId : value=cocoId}

        ## nsddata_stimuli_hdf5
        imgBrick = self.read_stimuli_hdf5()

        ## captsions and instances of COCO
        captions_dict, categories_dict = self.read_coco_annotation()

        ## ROIs
        self.read_ROIs()
        
        train_saved_dir_path = join_paths(self.subject_saved_dir_path, 'train')
        test_saved_dir_path = join_paths(self.subject_saved_dir_path, 'test')
        check_and_make_dirs(train_saved_dir_path)
        check_and_make_dirs(test_saved_dir_path)

        for session_id in responses['SESSION'].unique():
            response = responses[responses['SESSION'] == session_id].to_numpy()
            space_type, nii_data = self.read_betas(session_id=session_id)
            assert len(response) == len(nii_data), print(f'Number of responses and betas are not equal in session {session_id}.')
            
            if space_type == 'func1mm':
                for trial, fmri in tqdm(zip(response, nii_data), total=len(nii_data), desc=f'Processing {self.subj} session {session_id}', leave=True):
                    # correct trial
                    if trial[column_of_ISCORRECT] == 1:
                        run_id = int(trial[column_of_RUN])
                        trial_id = int(trial[column_of_TRIAL])
                        session_run_trial_string = f'session{str(session_id).zfill(2)}_run{str(run_id).zfill(2)}_trial{str(trial_id).zfill(2)}'
                        KID_73 = int(trial[column_of_73KID]) - 1 # 0-based index
                        # Train Set
                        if not KID_73 in sharedixs:
                            saved_path = join_paths(train_saved_dir_path, session_run_trial_string)
                        # Test Set
                        else:
                            saved_path = join_paths(test_saved_dir_path, session_run_trial_string)
                        check_and_make_dirs(saved_path)

                        # fMRI
                        save_nii_file(fmri, join_paths(saved_path, 'fmri.nii.gz'))
                        # image: BLIP-2 encodes via RGB
                        image = Image.fromarray(imgBrick[KID_73])
                        image.save(join_paths(saved_path, 'image.png'))

                        # caption and other infos
                        OLD_flag = int(trial[column_of_ISOLD]) # 0 was novel, 1 was old
                        captions_list = captions_dict[stim_info[KID_73]] # list[str], each image has several captions
                        category_list = categories_dict[stim_info[KID_73]] # list[dict[str, any]], [{'supercategory', 'name', 'area}]
                        # Encode image and caption via BLIP-2, get embedding whose dim=768
                        batch_size = len(captions_list)
                        BLIP2_Dataloader = DataLoader(BLIP2_Dataset(image, captions_list, vis_processors, txt_processors, device), batch_size=batch_size, shuffle=False)
                        for sample in BLIP2_Dataloader:
                            # dim=(batch_size, num of queries, 768) before Tensor.mean(dim=1)
                            # dim=(batch_size, 768) after Tensor.mean(dim=1), where batch_size = number of captions of this image
                            multimodal_embedding = torch.squeeze(BLIP2_Encoder_model.extract_features(sample).multimodal_embeds.mean(dim=1)).to('cpu')
                            image_embedding = torch.squeeze(BLIP2_Encoder_model.extract_features(sample, mode='image').image_embeds.mean(dim=1)).to('cpu')
                            caption_embedding = torch.squeeze(BLIP2_Encoder_model.extract_features(sample, mode='text').text_embeds.mean(dim=1)).to('cpu')

                        # Save the embeddings and information to npz file
                        save_npz_path = join_paths(saved_path, 'embedding.npz')
                        np.savez(
                            save_npz_path, 
                            image_embedding=image_embedding.numpy(), # numpy
                            captions_embedding=caption_embedding.numpy(), # numpy
                            multimodal_embedding=multimodal_embedding.numpy(), # numpy
                        )
                        
                        # Save the strings to json file
                        json_data = {
                            'isold' : 'novel' if OLD_flag == 0 else 'old', # str
                            'captions_list' : captions_list, # list[str]
                            'instances_category' : category_list
                        }
                        write_json_file(path = join_paths(saved_path, 'strings.json'), data = json_data)

                    # incorrect trial
                    else:
                        continue

            else:
                raise NotImplementedError(f'Space type: {space_type} is not supported.')
        print(f'{self.subj} has {len(os.listdir(train_saved_dir_path))} pairs in train set, {len(os.listdir(test_saved_dir_path))} pairs in test set.')

# make pairs of NSD
NSD_DATA(subj_id=1)  # 1 2 5 7      