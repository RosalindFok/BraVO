import os
import h5py
import time
import torch
import shutil
import scipy.io
# import warnings  
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from collections import Counter, defaultdict, namedtuple
from lavis.models.blip_diffusion_models.utils import preprocess_canny
# from SAM2.sam2.build_sam import build_sam2
# from SAM2.sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

from config import configs_dict
from models import device, num_workers, load_blip_models
from utils import NSD_dir_path, run_files_path, nsd_subject_saved_dir_path, sam2_ckpt_dir_path
from utils import join_paths, read_nii_file, save_nii_file, check_and_make_dirs, get_file_size, read_json_file, write_json_file, merge_dicts_if_no_conflict, get_items_in_list_via_substrs

os.environ['TOKENIZERS_PARALLELISM'] = 'false' 

DataPoint = namedtuple('DataPoint', ['index', 'image'])
class Dataset_for_BLIPs(Dataset):
    def __init__(self, path_dict : dict[int : dict[str : str]], vis_processors) -> None:
        super().__init__()
        self.path_dict = path_dict
        self.vis_processors = vis_processors

    def __getitem__(self, index) -> tuple[int, torch.Tensor]:
        image = Image.open(self.path_dict[index]['image'])
        image = self.vis_processors(image)
        image = torch.tensor(np.array(image))
        return DataPoint(index, image)

    def __len__(self) -> int:
        return len(self.path_dict)
                 

class NSD_DATA():
    def __init__(self, NSD_dir_path : str = NSD_dir_path, subj_id : int | str = None) -> None:
        super().__init__()
        # subj_id
        self.subj = f'subj{str(subj_id).zfill(2)}'
        
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
        self.nsddata_betas_ppdata_betas_dir_path = join_paths(NSD_dir_path, 'nsddata_betas', 'ppdata', self.subj, 'func1mm', 'betas_fithrf_GLMdenoise_RR')
        
        ## nsddata_stimuli
        # Info: https://cvnlab.slite.page/p/NKalgWd__F/Experiments#b44e32c0
        self.nsddata_stimuli_hdf5_file_path = join_paths(NSD_dir_path, 'nsddata_stimuli', 'stimuli', 'nsd', 'nsd_stimuli.hdf5')
       
        ## COCO annotation
        # Ref: https://cvnlab.slite.page/p/M3ZvPmfgU3/General-Information#1d5942f6
        self.coco_annotation_dir_path = join_paths(NSD_dir_path, 'nsddata_stimuli', 'stimuli', 'nsd', 'annotations')

        ## saved path of this subject
        self.subject_saved_dir_path = nsd_subject_saved_dir_path

    
    def read_behav_responses_tsv(self) -> pd.core.frame.DataFrame:
        start_time = time.time()
        data_frame = pd.read_csv(self.behav_responses_tsv_file_path, sep='\t', encoding='utf-8')
        subj_numpyINT64 = np.int64(self.subj[-1])
        assert (data_frame['SUBJECT'] == subj_numpyINT64).all(), f'Subject id in tsv file is not correct.' # subj 1~8
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

    def read_betas(self, session_id : int) -> tuple[str, np.ndarray]:
        start_time = time.time()
        # Info: https://cvnlab.slite.page/p/6CusMRYfk0/Functional-data-NSD#3e1740b1
        file_name = f'betas_session{str(session_id).zfill(2)}.nii.gz'
        file_path = join_paths(self.nsddata_betas_ppdata_betas_dir_path, file_name)
        header, data = read_nii_file(file_path) # dims=(145, 186, 148, 750); dtype=float64
        assert np.iinfo(np.int16).min <= np.min(data) and np.iinfo(np.int16).max >= np.max(data), 'Data range is not within int16 range.'
        data = data.astype(np.int16)
        data = np.transpose(data, (3, 0, 1, 2)) # dims=(750, 145, 186, 148)
        end_time = time.time()
        print(f'It took {end_time - start_time:.2f} seconds to read {file_path}.')
        return data
        
    def read_stimuli_hdf5(self) -> np.ndarray:
        start_time = time.time()
        with h5py.File(self.nsddata_stimuli_hdf5_file_path, 'r') as f:
            # imgBrick is 3 channels x 425 pixels x 425 pixels x 73,000 images and is in uint8 format. 
            # These images are shown on a gray background with RGB value (127,127,127).
            imgBrick = f['imgBrick'][:]
        end_time = time.time()
        print(f'It took {end_time - start_time:.2f} seconds to read {self.nsddata_stimuli_hdf5_file_path}.')
        return imgBrick
    
    def read_coco_annotation(self) -> tuple[dict[int, list[str]], dict[int, list[dict[str, any]]]]:
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
        del captions_train2017, captions_val2017, captions_train_annotations, captions_val_annotations, train_annotations, val_annotations

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
        del instances_train2017, instances_val2017, annotations_train2017, annotations_val2017, categories_train2017, categories_val2017, train_categories, val_categories
        
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
            assert len(roi_path_list) == 3, f'There are {len(roi_path_list)} ROIs for {tag}.'
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
            assert len(label_path_list) == 1, f'There should be only one label file for {tag}, the label_path_list is {label_path_list}.'
            label_path = label_path_list[0]
            label_tags_dict = {-1 : 'non-cortical voxels'} # {key=label_id : value=name}
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.replace('\n', '').split(' ')
                    line = [s.replace('\t', '') for s in line if s]
                    assert len(line) == 2, f'Invalid line: {line} of path = {label_path}'
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

    def make_pairs(self) -> dict[str : dict[int : dict[str : str]]]:
        """
        fMRI <--> image + text
        """        
        ## Path of recorded processed data paths
        processed_data_paths_json_path = join_paths(self.subject_saved_dir_path, 'processed_data_paths.json')
        if os.path.exists(processed_data_paths_json_path):
            path_dict = read_json_file(path=processed_data_paths_json_path)
            return path_dict

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
        
        ## Paths of train set and test set
        train_saved_dir_path = join_paths(self.subject_saved_dir_path, 'train')
        test_saved_dir_path = join_paths(self.subject_saved_dir_path, 'test')
        check_and_make_dirs(train_saved_dir_path)
        check_and_make_dirs(test_saved_dir_path)

        # Subj01, 02, 05, 07. Each subject has 40 sessions, each session has 750 trials.
        path_dict = {'train' : [], 'test' : []} 
        for session_id in responses['SESSION'].unique():
            response = responses[responses['SESSION'] == session_id].to_numpy()
            nii_data = self.read_betas(session_id=session_id)
            assert len(response) == len(nii_data), f'Number of responses and betas are not equal in session {session_id}.'
            for trial, fmri in tqdm(zip(response, nii_data), total=len(nii_data), desc=f'Processing {self.subj} session {session_id}', leave=True):
                # correct trial
                if trial[column_of_ISCORRECT] == 1:
                    run_id = int(trial[column_of_RUN])
                    trial_id = int(trial[column_of_TRIAL])
                    session_run_trial_string = f'session{str(session_id).zfill(2)}_run{str(run_id).zfill(2)}_trial{str(trial_id).zfill(2)}'
                    KID_73 = int(trial[column_of_73KID]) - 1 # 0-based index
                    image_array = imgBrick[KID_73].astype(np.uint8) # numpy.ndarray, shape=(425, 425, 3)

                    # Note: Split data into train and test sets based on whether the 73KID is part of the shared indices.
                    # Train Set
                    if not KID_73 in sharedixs:
                        saved_path = join_paths(train_saved_dir_path, session_run_trial_string)
                        path_dict['train'].append(saved_path)
                    # Test Set
                    else:
                        saved_path = join_paths(test_saved_dir_path, session_run_trial_string)
                        path_dict['test'].append(saved_path)
                    check_and_make_dirs(saved_path)
                    
                    # fMRI
                    fmri_path = join_paths(saved_path, 'fmri.nii.gz')
                    if not os.path.exists(fmri_path):
                        save_nii_file(fmri, fmri_path)

                    # image
                    image_path = join_paths(saved_path, 'image.png')
                    if not os.path.exists(image_path):
                        image_rgb = Image.fromarray(image_array).convert('RGB')
                        image_rgb.save(image_path)

                    # canny
                    canny_path = join_paths(saved_path, 'canny.png')
                    if not os.path.exists(canny_path):
                        canny_image = preprocess_canny(input_image=image_array, image_resolution=image_array.shape[0], 
                                                       low_threshold=100, high_threshold=200
                                                    )
                        canny = np.array(canny_image)
                        if not np.max(canny) > np.min(canny):
                            canny_image = preprocess_canny(input_image=image_array, image_resolution=image_array.shape[0], 
                                                           low_threshold=np.min(canny)//2, high_threshold=np.max(canny)//2
                                                        )
                        canny = np.array(canny_image)
                        assert np.max(canny) > np.min(canny), f'Canny image is all black in path={saved_path}!'
                        canny_image.save(canny_path)

                    # strings
                    strings_path = join_paths(saved_path, 'strings.json')
                    if not os.path.exists(strings_path):
                        # captions and categories from COCO annotation
                        captions_list = captions_dict[stim_info[KID_73]] # list[str], each image has several captions
                        category_list = categories_dict[stim_info[KID_73]] # list[dict[str, any]], [{'supercategory', 'name', 'area}]
                        # string: describe the number of each category in the image
                        element_counts = Counter([category['name'] for category in category_list])
                        category_string = 'There are ' + ', '.join(f'{count} {element}' for element, count in element_counts.items()) + ' in this image.'
                        # select the category with the biggest area in sum
                        area_of_each_category = defaultdict(float) 
                        for category in category_list:
                            area_of_each_category[category['name']] += category['area']  
                        selected_category = max(area_of_each_category, key=lambda k: area_of_each_category[k])
                        # save the strings to json file
                        json_data = {
                            'coco_captions' : captions_list, # list[str]
                            'coco_category' : category_list, # list[dict[str, any]]
                            'selected_category' : selected_category,   # str
                            'category_string' : category_string # str
                        }
                        write_json_file(path=strings_path, data=json_data)
                
                # incorrect trial
                else:
                    continue
        
        # Reorganize the path_dict
        for key, item in path_dict.items():
            files_dict = {} # {index : {keyword : path}}
            for index, trail_dir_path in enumerate(item):
                files = os.listdir(trail_dir_path)
                files_dict[index] = {file.split('.')[0] : join_paths(trail_dir_path, file) for file in files}
            path_dict[key] = files_dict
        write_json_file(path=processed_data_paths_json_path, data=path_dict)
        # Check if the number of samples in json file is equal to the number of directory
        path_dict = read_json_file(path=processed_data_paths_json_path)
        assert len(path_dict['train']) == len(os.listdir(train_saved_dir_path)), f"Number of train samples in json={len(path_dict['train'])} is not equal to the number of directory={len(os.listdir(train_saved_dir_path))}."
        assert len(path_dict['test']) == len(os.listdir(test_saved_dir_path)), f"Number of test samples in json={len(path_dict['test'])} is not equal to the number of directory={len(os.listdir(test_saved_dir_path))}."
        print(f'{self.subj} has {len(os.listdir(train_saved_dir_path))} pairs in train set, {len(os.listdir(test_saved_dir_path))} pairs in test set.')
        return path_dict

    def blip2_process(self, path_dict : dict[str : dict[str : dict[str : str]]]) -> None:
        # Save the results processed by BLIPs
        blips_output_dir_path = join_paths(self.subject_saved_dir_path, 'blips_output')
        check_and_make_dirs(blips_output_dir_path)

        # Convert the keys of path_dict from str to int
        path_dict = {key: {int(k) : v for k, v in value.items()} for key, value in path_dict.items()}  
        num_train, num_test = len(path_dict['train']), len(path_dict['test'])
        
        ## Generate captions for each image
        blip2t5_generated_captions_of_train_set_path = join_paths(blips_output_dir_path, 'blip2t5_generated_captions_of_train_set.json')
        blip2t5_generated_captions_of_test_set_path  = join_paths(blips_output_dir_path, 'blip2t5_generated_captions_of_test_set.json')
        need_caption_train = (  
            os.path.exists(blip2t5_generated_captions_of_train_set_path) and  
            len(read_json_file(blip2t5_generated_captions_of_train_set_path)) == num_train  
        )
        need_caption_test = (  
            os.path.exists(blip2t5_generated_captions_of_test_set_path) and  
            len(read_json_file(blip2t5_generated_captions_of_test_set_path)) == num_test  
        )

        if not (need_caption_train and need_caption_test):
            # Load blip2 model
            blip2t5_model, blip2t5_vis_processors, _ = load_blip_models(mode='caption') 

            batch_size = 12
            prompt = configs_dict['blip_caption']['prompt']
            dataset_queue, tag_queue, saved_path_queue, number_queue = [], [], [], []
            if not need_caption_test:
                dataset_queue.append(Dataset_for_BLIPs(path_dict=path_dict['test'],  vis_processors=blip2t5_vis_processors['eval']))
                tag_queue.append('Test')
                saved_path_queue.append(blip2t5_generated_captions_of_test_set_path)
                number_queue.append(num_test)
            if not need_caption_train:
                dataset_queue.append(Dataset_for_BLIPs(path_dict=path_dict['train'], vis_processors=blip2t5_vis_processors['eval']))
                tag_queue.append('Train')
                saved_path_queue.append(blip2t5_generated_captions_of_train_set_path)
                number_queue.append(num_train)
            captions_set = {} # {index : caption}
            for dataset, tag, saved_path, number in zip(dataset_queue, tag_queue, saved_path_queue, number_queue):
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
                for batches in tqdm(dataloader, desc=f'{tag} set', leave=True):
                    indices = batches.index.to('cpu').numpy()
                    images = batches.image.to(device) # torch.Size([batch_size, 425, 425, 3])
                    output_text = blip2t5_model.generate({'image' : images, 'prompt' : prompt},
                                                          max_length=100, min_length=30)
                    for index, text in zip(indices, output_text):
                        index = int(index)
                        captions_set[index] = text
                assert len(captions_set) == number, f'The length of generated captions of {tag.lower()} set is {len(captions_set)}, which should be {number}.'
                write_json_file(path=saved_path, data=captions_set)

            # Delete the loaded model
            del blip2t5_model, blip2t5_vis_processors
        
        ## Generate embedding for each pair
        blipdiffusion_generated_embeddings_of_train_set_path = join_paths(blips_output_dir_path, 'blipdiffusion_generated_embeddings_of_train.hdf5')
        blipdiffusion_generated_embeddings_of_test_set_path  = join_paths(blips_output_dir_path, 'blipdiffusion_generated_embeddings_of_test.hdf5')

        def __check_hdf5_file__(path : str, target_length : int) -> bool:
            if os.path.exists(path):
                try:
                    with h5py.File(path, 'r') as hdf5_file:
                        if len(hdf5_file.keys()) == target_length:
                            return True
                except Exception as e:
                    os.remove(path)
            return False

        need_embedding_train = __check_hdf5_file__(path=blipdiffusion_generated_embeddings_of_train_set_path, target_length=num_train)
        need_embedding_test  = __check_hdf5_file__(path=blipdiffusion_generated_embeddings_of_test_set_path , target_length=num_test)
        uncond_embedding_path = join_paths(blips_output_dir_path, 'uncond_embedding.npy')
        causal_attention_mask_path = join_paths(blips_output_dir_path, 'causal_attention_mask.npy')
        all_strings_train_path = join_paths(self.subject_saved_dir_path,f'all_strings_path_in_train.json')
        all_strings_test_path  = join_paths(self.subject_saved_dir_path,f'all_strings_path_in_test.json')
        run_files_dict = {
            'train' : {
                'hdf5' : blipdiffusion_generated_embeddings_of_train_set_path,
                'json' : all_strings_train_path
            },
            'test'  : {
                'hdf5' : blipdiffusion_generated_embeddings_of_test_set_path,
                'json' : all_strings_test_path
            },
            'uncond_embedding_path' : uncond_embedding_path,
            'causal_attention_mask_path' : causal_attention_mask_path
        }
        write_json_file(path=run_files_path, data=run_files_dict)

        if not (need_embedding_train and need_embedding_test):
            # Load blip2 model
            blip_diffusion_model, bd_vis_processors, bd_txt_processors = load_blip_models(mode='diffusion')
            # save_uncond_embedding
            uncond_embedding = blip_diffusion_model.generate_uncond_embedding(neg_prompt=configs_dict['blip_diffusion']['negative_prompt'])
            uncond_embedding = uncond_embedding.cpu().numpy()
            np.save(uncond_embedding_path, uncond_embedding)
            assert uncond_embedding.shape == (1, 77, 768), f'uncond_embedding.shape={uncond_embedding.shape} is not (1, 77, 768).'

            files_queue, tag_queue, hdf5_path_queue, captions_queue, json_path_queue = [], [], [], [], []
            if not need_embedding_test:
                files_queue.append(path_dict['test'])
                tag_queue.append('Test')
                hdf5_path_queue.append(blipdiffusion_generated_embeddings_of_test_set_path)
                captions_queue.append(read_json_file(blip2t5_generated_captions_of_test_set_path))
                json_path_queue.append(all_strings_test_path)
            if not need_embedding_train:
                files_queue.append(path_dict['train'])
                tag_queue.append('Train')
                hdf5_path_queue.append(blipdiffusion_generated_embeddings_of_train_set_path)
                captions_queue.append(read_json_file(blip2t5_generated_captions_of_train_set_path))
                json_path_queue.append(all_strings_train_path)
            for files, tag, hdf5_path, captions, json_path in zip(files_queue, tag_queue, hdf5_path_queue, captions_queue, json_path_queue):
                captions = {int(k) : v for k, v in captions.items()} 
                all_strings_path = {} # {index : path}
                with h5py.File(hdf5_path, 'w') as hdf5_file:
                    for index, files_dict in tqdm(files.items(), desc=f'{tag} set', leave=True):
                        hdf5_group = hdf5_file.create_group(name=str(index))
                        image_rgb = Image.open(files_dict['image']).convert('RGB')
                        _, fmri = read_nii_file(files_dict['fmri'])
                        strings_path = files_dict['strings']
                        strings = read_json_file(strings_path)
                        caption = captions[index]
                        strings['blip_caption'] = caption
                        write_json_file(path=strings_path, data=strings)
                        all_strings_path[index] = strings_path
                        category_string = strings['category_string']

                        cond_image = bd_vis_processors['eval'](image_rgb).unsqueeze(0).to(device)
                        category_string = bd_txt_processors['eval'](category_string)
                        sample = {
                            'cond_images'  : cond_image,
                            'prompt'       : [bd_txt_processors['eval'](strings['category_string']+strings['blip_caption'])],
                            'cond_subject' : category_string,
                            'tgt_subject'  : category_string
                        }
                        hidden_states, causal_attention_mask = blip_diffusion_model.generate_embedding(samples=sample)
                        assert hidden_states.shape == (1, 77, 768), f'embedding shape is {hidden_states.shape}, not (1, 77, 768).'
                        assert causal_attention_mask.shape == (1, 1, 77, 77), f'causal_attention_mask shape is {causal_attention_mask.shape}, not (1, 1, 77, 77).'
                        hidden_states = hidden_states.cpu().numpy()
                        causal_attention_mask = causal_attention_mask.cpu().numpy()
                        if not os.path.exists(causal_attention_mask_path):
                            np.save(file=causal_attention_mask_path, arr=causal_attention_mask)
                        image = np.array(image_rgb)
                        for name, data in zip(['image', 'fmri', 'hidden_states'], 
                                              [ image ,  fmri ,  hidden_states ]): # shape: (425, 425, 3),(145, 186, 148),(1, 77, 768)
                            hdf5_dataset = hdf5_group.create_dataset(name=name, shape=data.shape, dtype=data.dtype)
                            hdf5_dataset[:] = data
                write_json_file(path=json_path, data=all_strings_path)        
                print(f'{hdf5_path} is created, the size of it is {get_file_size(hdf5_path)}')

            # Delete the loaded model
            del blip_diffusion_model, bd_vis_processors, bd_txt_processors   

        # ## Load SAM2 model
        # checkpoint = join_paths(sam2_ckpt_dir_path, 'sam2_hiera_large.pt')
        # sam2 = build_sam2('sam2_hiera_l.yaml', checkpoint, device=device, apply_postprocessing=False)
        # mask_generator = SAM2AutomaticMaskGenerator(model=sam2, points_per_side=64, points_per_batch=128,
        #                                             pred_iou_thresh=0.7, stability_score_thresh=0.92,
        #                                             stability_score_offset=0.7, crop_n_layers=1,
        #                                             box_nms_thresh=0.7, crop_n_points_downscale_factor=2,
        #                                             min_mask_region_area=25.0, use_m2m=True
        #                                         )

# make pairs of NSD
if __name__ == '__main__':
    nsd_data = NSD_DATA(subj_id=configs_dict['subj_id'])
    path_dict = nsd_data.make_pairs()
    nsd_data.blip2_process(path_dict=path_dict)