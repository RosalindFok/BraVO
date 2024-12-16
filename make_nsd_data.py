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
from torch.utils.data import Dataset, DataLoader
from collections import Counter, defaultdict, namedtuple

from config import configs_dict
from models import device, num_workers, load_blip_models
from utils import NSD_dir_path, run_files_path, nsd_subject_saved_dir_path
from utils import join_paths, read_nii_file, save_nii_file, read_json_file, write_json_file, merge_dicts_if_no_conflict, get_items_in_list_via_substrs, BLIP_Prior_Tools

os.environ['TOKENIZERS_PARALLELISM'] = 'false' 

DataPoint = namedtuple('DataPoint', ['dir_path', 'image'])
class Dataset_for_BLIPs(Dataset):
    """
    Dataset for BLIP2t5 model, which generates captions for images.
    """
    def __init__(self, path_list : list[str], vis_processors) -> None:
        super().__init__()
        self.path_list = path_list
        self.vis_processors = vis_processors

    def __getitem__(self, index) -> tuple[str, torch.Tensor]:
        dir_path = self.path_list[index]
        image = Image.open(join_paths(dir_path, 'image.png')) # shape=[425,425,3]
        image = self.vis_processors(image) # type=torch.Tensor, shape=[3, 364, 364]
        return DataPoint(dir_path, image)

    def __len__(self) -> int:
        return len(self.path_list)
                 

class NSD_DATA():
    def __init__(self, NSD_dir_path : str = NSD_dir_path, subj_id : int | str = None) -> None:
        super().__init__()
        # subj_id
        self.subj = f'subj{str(subj_id).zfill(2)}'
        
        self.functional_space = configs_dict['functional_space']

        ## nsddata
        self.nsddata_dir_path = join_paths(NSD_dir_path, 'nsddata')
        self.nsddata_ppdata_dir_path = join_paths(self.nsddata_dir_path, 'ppdata')
        # Info: https://cvnlab.slite.page/p/fRv4lz5V2F/Behavioral-data#2bdd55ef
        self.behav_responses_tsv_file_path = join_paths(self.nsddata_ppdata_dir_path, self.subj, 'behav', 'responses.tsv')
        # https://cvnlab.slite.page/p/X_7BBMgghj/ROIs#c5518e3e
        self.roi_files_path = join_paths(self.nsddata_ppdata_dir_path, self.subj, self.functional_space, 'roi')
        # https://cvnlab.slite.page/p/X_7BBMgghj/ROIs#2da19afb
        self.labels_path = join_paths(self.nsddata_dir_path, 'freesurfer', self.subj, 'label')
        # https://cvnlab.slite.page/p/X_7BBMgghj/ROIs#65b75445
        self.templates_path = join_paths(self.nsddata_dir_path, 'templates')
        # Info: https://cvnlab.slite.page/p/NKalgWd__F/Experiments#b0ea56ab
        self.expdesign_mat_file_path = join_paths(self.nsddata_dir_path, 'experiments', 'nsd', 'nsd_expdesign.mat')
        # Info: https://cvnlab.slite.page/p/NKalgWd__F/Experiments#bf18f984
        self.stim_info_csv_file_path = join_paths(self.nsddata_dir_path, 'experiments', 'nsd', 'nsd_stim_info_merged.csv')

        ## nsddata_betas
        # Info: https://cvnlab.slite.page/p/6CusMRYfk0/Functional-data-NSD#035bbb1e
        self.nsddata_betas_ppdata_betas_dir_path = join_paths(NSD_dir_path, 'nsddata_betas', 'ppdata', self.subj, self.functional_space, 'betas_fithrf_GLMdenoise_RR')
        
        ## nsddata_stimuli
        # Info: https://cvnlab.slite.page/p/NKalgWd__F/Experiments#b44e32c0
        self.nsddata_stimuli_hdf5_file_path = join_paths(NSD_dir_path, 'nsddata_stimuli', 'stimuli', 'nsd', 'nsd_stimuli.hdf5')
       
        ## COCO annotation
        # Ref: https://cvnlab.slite.page/p/M3ZvPmfgU3/General-Information#1d5942f6
        self.coco_annotation_dir_path = join_paths(NSD_dir_path, 'nsddata_stimuli', 'stimuli', 'nsd', 'annotations')

    def read_behav_responses_tsv(self) -> pd.core.frame.DataFrame:
        """  
        Reads behavioral response data from a tab-separated values (TSV) file and returns it as a Pandas DataFrame.  

        The function performs the following steps:  
        1. Reads the TSV file into a Pandas DataFrame.  
        2. Validates that the `SUBJECT` column in the file matches the subject ID provided in the instance variable.  
        3. Drops unnecessary columns from the DataFrame to retain only relevant data for further analysis.  

        Returns:  
            pd.core.frame.DataFrame: A Pandas DataFrame containing the cleaned behavioral response data.  
        """  
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
        """  
        Reads the experimental design data from a MATLAB (.mat) file and returns its contents as a dictionary.  

        The .mat file contains experimental design information, which is loaded using the `scipy.io.loadmat`   
        function. The data is returned as a dictionary where keys are variable names from the .mat file, and   
        values are the corresponding data structures (e.g., arrays, matrices, or other objects).  

        Returns:  
            dict[str, any]: A dictionary containing the contents of the .mat file. The keys are strings   
                            representing variable names, and the values are the corresponding data.  
        """  
        start_time = time.time()
        mat_contents = scipy.io.loadmat(self.expdesign_mat_file_path)
        end_time = time.time()
        print(f'It took {end_time - start_time:.2f} seconds to read {self.expdesign_mat_file_path}.')
        return mat_contents
    
    def read_stim_info_csv(self) -> dict[int, int]:
        """  
        Reads the stimulus information CSV file and returns a mapping of NSD image IDs to COCO IDs.  

        The CSV file contains information about the images used in the NSD (Natural Scenes Dataset) experiment.   
        Each row in the file corresponds to an image, with the following key columns:  
        - `nsdId`: A 0-based index representing the image ID in the NSD experiment.   
                   This is the index into the full set of 73,000 images used in the experiment.  
        - `cocoId`: The ID assigned to the image in the COCO (Common Objects in Context) database.  

        This function reads the CSV file, extracts the `nsdId` and `cocoId` columns, and creates a dictionary   
        mapping `nsdId` (keys) to `cocoId` (values).  

        Returns:  
            dict[int, int]: A dictionary where the keys are `nsdId` (0-based) and the values are `cocoId`.  
        """  
        start_time = time.time()
        data_frame = pd.read_csv(self.stim_info_csv_file_path)
        # cocoId: is the ID number assigned to this image in the COCO database.
        # nsdId: is the 0-based index of the image into the full set of 73k images used in the NSD experiment. Values are the same as column 1. (Note that in some other cases, 73k IDs are specified as 1-based. Here the IDs are specified as 0-based.)
        end_time = time.time()
        print(f'It took {end_time - start_time:.2f} seconds to read {self.stim_info_csv_file_path}.')
        return dict(zip(data_frame['nsdId'], data_frame['cocoId']))

    def read_betas(self, session_id : int) -> np.array:
        """  
        Reads beta weight data for a specified session from a preprocessed NIfTI file and returns it as a NumPy array.  

        Beta weights are stored in NIfTI files, where each file corresponds to a specific session.   
        The function reads the file, ensures the data values are within the range of int16,   
        converts the data to int16 format, and transposes it to rearrange its dimensions.  

        Args:  
            session_id (int): The session ID for which the beta weights need to be read.   
                              The session ID is zero-padded to two digits (e.g., 1 -> '01')   
                              to match the file naming convention.  

        Returns:  
            np.array: A NumPy array containing the beta weight data with dimensions   
                      transposed to (timepoints, x, y, z).  
        """  
        start_time = time.time()
        # Info: https://cvnlab.slite.page/p/6CusMRYfk0/Functional-data-NSD#3e1740b1
        file_name = f'betas_session{str(session_id).zfill(2)}.nii.gz'
        file_path = join_paths(self.nsddata_betas_ppdata_betas_dir_path, file_name)
        _, data = read_nii_file(file_path) 
        assert np.iinfo(np.int16).min <= np.min(data) and np.iinfo(np.int16).max >= np.max(data), 'Data range is not within int16 range.'
        data = data.astype(np.int16)
        data = np.transpose(data, (3, 0, 1, 2)) 
        end_time = time.time()
        print(f'It took {end_time - start_time:.2f} seconds to read {file_path}.')
        return data
        
    def read_stimuli_hdf5(self) -> np.array:
        """  
        Reads the stimuli data from an HDF5 file and returns it as a NumPy array.  

        This function reads the dataset named 'imgBrick' from the specified HDF5 file.  
        The dataset contains image data in the format of 3 channels (RGB) x 425 pixels x 425 pixels x 73,000 images.  
        The images are stored in uint8 format and are displayed on a gray background with RGB values (127, 127, 127).  

        Returns:  
            np.array: A NumPy array containing the image data from the HDF5 file.  
        """  
        start_time = time.time()
        with h5py.File(self.nsddata_stimuli_hdf5_file_path, 'r') as f:
            # imgBrick is 3 channels x 425 pixels x 425 pixels x 73,000 images and is in uint8 format. 
            # These images are shown on a gray background with RGB value (127,127,127).
            imgBrick = f['imgBrick'][:]
        end_time = time.time()
        print(f'It took {end_time - start_time:.2f} seconds to read {self.nsddata_stimuli_hdf5_file_path}.')
        return imgBrick
    
    def read_coco_annotation(self) -> tuple[dict[int, list[str]], dict[int, list[dict[str, any]]]]:
        """  
        Reads and processes COCO annotation files to extract captions and category information.  

        This function processes the COCO dataset's caption and instance annotations, extracting:  
        - Captions for each image (multiple captions per image are possible).  
        - Category information for each image, including supercategories, names, and areas of objects.  

        Returns:  
            tuple: A tuple containing two dictionaries:  
                - captions_dict (dict[int, list[str]]): Maps image IDs to a list of captions.  
                - categories_dict (dict[int, list[dict[str, any]]]): Maps image IDs to a list of category details.  
                    Each category detail includes:  
                        - 'supercategory': The supercategory of the object.  
                        - 'name': The name of the object category.  
                        - 'area': The area of the object in the image.  
        """  
        
        def __extract_captions__(captions_annotations : list[dict[str, any]]) -> dict[int, list[str]]:
            """  
            Extracts captions for each image from the COCO annotations.  

            Args:  
                captions_annotations (list[dict[str, any]]): A list of caption annotations, where each annotation  
                    contains 'image_id' (int) and 'caption' (str).  

            Returns:  
                dict[int, list[str]]: A dictionary mapping image IDs to a list of captions.  
            """
            annotations = {} # {key=id : value=[caption1, caption2, ...]}
            # some pictures have multiple captions
            for ca in captions_annotations:
                if not ca['image_id'] in annotations:
                    annotations[ca['image_id']] = [ca['caption']] 
                else:
                    annotations[ca['image_id']].append(ca['caption'])
            return annotations
        
        def __extract_categories__(annotations_list : list[dict[str, any]], categories_list : list[dict[str, any]]) -> dict[int, list[dict[str, any]]]:
            """  
            Extracts category details for each image from the COCO instance annotations.  

            Args:  
                annotations_list (list[dict[str, any]]): A list of instance annotations, where each annotation  
                    contains 'image_id' (int), 'category_id' (int), and 'area' (float).  
                categories_list (list[dict[str, any]]): A list of category definitions, where each category contains  
                    'id' (int), 'supercategory' (str), and 'name' (str).  

            Returns:  
                dict[int, list[dict[str, any]]]: A dictionary mapping image IDs to a list of category details.  
                    Each category detail includes:  
                        - 'supercategory': The supercategory of the object.  
                        - 'name': The name of the object category.  
                        - 'area': The area of the object in the image.  
            """ 
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
    
    def read_ROIs(self) -> str:
        start_time = time.time()
        # saved path for ROIs
        saved_rois_path = join_paths(nsd_subject_saved_dir_path, 'ROIs', self.functional_space)
        os.makedirs(saved_rois_path, exist_ok=True)

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
            os.makedirs(saved_path, exist_ok=True)

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
            return None

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
        return saved_rois_path

    def make_pairs(self) -> None:
        """
        Deletes trials where ISOLD=1 (old images) or ISCORRECT=0 (incorrect responses).  
        Splits data into train and test sets:
            - Shared trials -> test set  
            - Remaining trials -> train set  
        Final pairs include: fMRI data, image, and descriptive strings. 

        Returns:  
            dict: A dictionary containing paths to paired data for training and testing sets.  
                The structure is {set_type: {index: {keyword: path}}}.  
        """  
        
        # if done, return
        method_done_path = join_paths(nsd_subject_saved_dir_path, '_'.join([self.make_pairs.__name__, self.functional_space, 'done']))
        if os.path.exists(method_done_path):
            return None
        
        ## ROIs
        saved_rois_path = self.read_ROIs()
        key = 'ROIs'
        value = {self.functional_space : saved_rois_path}
        if not os.path.exists(run_files_path):
            write_json_file(path=run_files_path, data={key : value})
        else:
            old_dict = read_json_file(path=run_files_path)
            if key in old_dict:
                old_dict[key].update(value)
            else:
                old_dict[key] = value
            write_json_file(path=run_files_path, data=old_dict) 

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
        
        ## Paths of train set and test set
        train_saved_dir_path = join_paths(nsd_subject_saved_dir_path, 'train')
        test_saved_dir_path  = join_paths(nsd_subject_saved_dir_path, 'test')
        os.makedirs(train_saved_dir_path, exist_ok=True)
        os.makedirs(test_saved_dir_path, exist_ok=True)
        write_json_file(path=run_files_path, data={**read_json_file(path=run_files_path), 
                                                   **{'train': train_saved_dir_path, 
                                                      'test' : test_saved_dir_path}})

        # Subj01, 02, 05, 07. Each subject has 40 sessions, each session has 750 trials.
        for session_id in responses['SESSION'].unique():
            response = responses[responses['SESSION'] == session_id].to_numpy()
            # func1mm    shape = (750, 145, 186, 148)
            # func1pt8mm shape = (750, 81 , 104, 83 )
            nii_data = self.read_betas(session_id=session_id)
            assert len(response) == len(nii_data), f'Number of responses and betas are not equal in session {session_id}.'
            for trial, fmri in tqdm(zip(response, nii_data), total=len(nii_data), desc=f'Processing {self.subj} session {session_id}', leave=True):
                # correct trial and image is novel
                if trial[column_of_ISCORRECT] == 1 and trial[column_of_ISOLD] == 0:
                    run_id = int(trial[column_of_RUN])
                    trial_id = int(trial[column_of_TRIAL])
                    session_run_trial_string = f'session{str(session_id).zfill(2)}_run{str(run_id).zfill(2)}_trial{str(trial_id).zfill(2)}'
                    KID_73 = int(trial[column_of_73KID]) - 1 # 0-based index
                    image_array = imgBrick[KID_73].astype(np.uint8) # numpy.ndarray, shape=(425, 425, 3)

                    # Note: Split data into train and test sets based on whether the 73KID is part of the shared indices.
                    # Train Set
                    if not KID_73 in sharedixs:
                        saved_path = join_paths(train_saved_dir_path, session_run_trial_string)
                    # Test Set
                    else:
                        saved_path = join_paths(test_saved_dir_path, session_run_trial_string)
                    os.makedirs(saved_path, exist_ok=True)
                    
                    # fMRI
                    fmri_path = join_paths(saved_path, f'{self.functional_space}_fmri.nii.gz')
                    if not os.path.exists(fmri_path):
                        save_nii_file(fmri, fmri_path)

                    # image
                    image_path = join_paths(saved_path, 'image.png')
                    if not os.path.exists(image_path):
                        image_rgb = Image.fromarray(image_array).convert('RGB')
                        image_rgb.save(image_path)

                    # strings
                    strings_path = join_paths(saved_path, 'strings.json')
                    if not os.path.exists(strings_path):
                        # captions and categories from COCO annotation
                        captions_list = captions_dict[stim_info[KID_73]] # list[str], each image has several captions
                        category_list = categories_dict[stim_info[KID_73]] # list[dict[str, any]], [{'supercategory':str, 'name':str, 'area':int}]
                        # string: describe the number of each category in the image
                        element_counts = Counter([category['name'] for category in category_list])
                        # category_string is like: 1 cow, 2 dog, 3 cat. 
                        category_string = ', '.join(f'{count} {element}' for element, count in element_counts.items())+'. '
                        # select the category with the biggest area in sum
                        area_of_each_category = defaultdict(float) 
                        for category in category_list:
                            area_of_each_category[category['name']] += category['area']  
                        # save the strings to json file
                        json_data = {
                            'coco_captions' : captions_list, # list[str]
                            'coco_category' : category_list, # list[dict[str, any]]
                            'category_string' : category_string # str
                        }
                        write_json_file(path=strings_path, data=json_data)
                
                # incorrect trial or image is old
                else:
                    continue
        
        # write done
        with open(method_done_path, 'w') as f:
            f.write('done')
        
        return None

    def blip2_process(self) -> None:
        """  
        Generates image_embeddings and captions using the BLIP2 model for train and test datasets.  
        """ 
        # if done, return
        method_done_path = join_paths(nsd_subject_saved_dir_path, '_'.join([self.blip2_process.__name__, 'done']))
        if os.path.exists(method_done_path):
            return
        
        run_files = read_json_file(run_files_path)
        trial_path_list = [join_paths(run_files[tag], d) for tag in ['test', 'train'] for d in os.listdir(run_files[tag])]

        # Load blip2 model
        blip2t5_model, blip2t5_vis_processors, _ = load_blip_models(mode='caption') 
        dataset = Dataset_for_BLIPs(path_list=trial_path_list,  vis_processors=blip2t5_vis_processors['eval'])
        batch_size = 12
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        for batches in tqdm(dataloader, desc=f'BLIP2t5 processing', leave=True):
            dir_paths = batches.dir_path
            images = batches.image.to(device)
            # image_embeds.shape=[bs, 677, 1408]
            image_embeds = blip2t5_model.generate_image_embeddings(images=images)
            # prompt_embeddings.shape=[bs, 5, 2048], prompt_attentions.shape=[bs, 5]
            prompt_embeddings, prompt_attentions = blip2t5_model.generate_default_prompt_embeddings(batch_size=image_embeds.size(0), device=device) # size(0)<=batch_size
            output_texts = blip2t5_model.generate_captions_via_embedding(
                image_embeds=image_embeds,
                prompt_embeddings=prompt_embeddings,
                prompt_attentions=prompt_attentions,
                max_length=configs_dict['blip2']['max_length'], 
                min_length=configs_dict['blip2']['min_length']
            )
            # save image embeddings and captions
            image_embeds = image_embeds.cpu().numpy()
            for dir_path, img_embed, caption in zip(dir_paths, image_embeds, output_texts):
                np.save(join_paths(dir_path, 'blip2_image_embedding.npy'), img_embed)
                strings_json_path = join_paths(dir_path, 'strings.json')
                strings = read_json_file(strings_json_path)
                strings['blip2_caption'] = caption
                write_json_file(strings_json_path, strings)

        # save fixed embeddsings: prompt_embedding, image_attention, prompt_attention
        blip2_output_fixed_dir_path = join_paths(nsd_subject_saved_dir_path, 'blip2_output_fixed')
        os.makedirs(blip2_output_fixed_dir_path, exist_ok=True)
        for name, tensor in zip(['prompt_embedding', 'prompt_attention'],
                                [ prompt_embeddings,  prompt_attentions]):
            file_path = join_paths(blip2_output_fixed_dir_path, f'{name}.npy')
            if not os.path.exists(file_path):
                array = tensor.float().cpu().numpy()[0]
                np.save(file_path, array)
        
        # write blip2_output_fixed_dir_path into run_files
        run_files['blip2_output_fixed'] = blip2_output_fixed_dir_path
        write_json_file(run_files_path, run_files)

        # write done
        with open(method_done_path, 'w') as f:
            f.write('all_done')
        
        # delete loaded models
        del blip2t5_model, blip2t5_vis_processors
        
        return None

    def blipdiffusion_process(self) -> None:
        """
        Generates image_embeddings and caption_embedings using the BLIP Diffusion Encoder for train and test datasets.  
        """

        # if done, return
        method_done_path = join_paths(nsd_subject_saved_dir_path, '_'.join([self.blipdiffusion_process.__name__, 'done']))
        if os.path.exists(method_done_path):
            return

        run_files = read_json_file(run_files_path)
        trial_path_list = [join_paths(run_files[tag], d) for tag in ['test', 'train'] for d in os.listdir(run_files[tag])]
        
        #  Load blip diffusion model
        blip_diffusion_model, bd_vis_processors, bd_txt_processors = load_blip_models(mode='diffusion')
        
        for trial_path in tqdm(trial_path_list, desc=f'BLIP Diffusion processing', leave=True):
            # image
            image = Image.open(join_paths(trial_path, 'image.png')).convert('RGB')
            image = bd_vis_processors['eval'](image).unsqueeze(0).to(device)
            # caption
            strings = read_json_file(join_paths(trial_path, 'strings.json'))
            blip2_caption = strings['blip2_caption']
            blip2_caption = [bd_txt_processors['eval'](blip2_caption)]
            category = strings['category_string']
            category = bd_txt_processors['eval'](category)
            # generate
            sample = {
                        'cond_images'  : image,
                        'prompt'       : blip2_caption,
                        'cond_subject' : category,
                        'tgt_subject'  : category,
            }
            hidden_states, position_embeddings, causal_attention_mask = blip_diffusion_model.generate_embedding(samples=sample)
            assert hidden_states.shape == (1, 77, 768), f'embedding shape={hidden_states.shape} != (1, 77, 768).'
            assert position_embeddings.shape == (1, 77, 768), f'position_embeddings shape={position_embeddings.shape} != (1, 77, 768).'
            assert causal_attention_mask.shape == (1, 1, 77, 77), f'causal_attention_mask shape={causal_attention_mask.shape} != (1, 1, 77, 77).'
            # 1. hidden_states - position_embeddings
            hidden_states -= position_embeddings
            # 2. split hidden_states into: image embedding, caption embedding, prefix, suffix
            prefix, image_embedding, caption_embedding, suffix = BLIP_Prior_Tools.split_hidden_states(hidden_states)
            # save image_embedding and caption_embedding
            np.save(join_paths(trial_path, 'blipdiffusion_image_embedding.npy'), image_embedding.cpu().numpy())
            np.save(join_paths(trial_path, 'blipdiffusion_caption_embedding.npy'), caption_embedding.cpu().numpy())

        # Save fixed embedding: uncond_embedding, position_embeddings, causal_attention_mask, prefix, suffix
        blipdiffusion_output_fixed_dir_path = join_paths(nsd_subject_saved_dir_path, 'blipdiffusion_output_fixed')
        os.makedirs(blipdiffusion_output_fixed_dir_path, exist_ok=True)
        # uncond_embedding: negative prompt
        uncond_embedding = blip_diffusion_model.generate_uncond_embedding(neg_prompt=configs_dict['blip_diffusion']['negative_prompt'])
        assert uncond_embedding.shape == (1, 77, 768), f'uncond_embedding.shape={uncond_embedding.shape} is not (1, 77, 768).'
        # save as npy
        for name, tensor in zip(['position_embeddings', 'causal_attention_mask', 'prefix', 'suffix', 'uncond_embedding'], 
                                [ position_embeddings,   causal_attention_mask,   prefix,   suffix,   uncond_embedding]):
            array = tensor.cpu().numpy()
            np.save(join_paths(blipdiffusion_output_fixed_dir_path, f'{name}.npy'), array)

        # write blipdiffusion_output_fixed_dir_path into run_files
        run_files['blipdiffusion_output_fixed'] = blipdiffusion_output_fixed_dir_path
        write_json_file(path=run_files_path, data=run_files)

        # write done
        with open(method_done_path, 'w') as f:
            f.write('done')

        # delete loaded models
        del blip_diffusion_model, bd_vis_processors, bd_txt_processors

        return None

# make pairs of NSD
if __name__ == '__main__':
    nsd_data = NSD_DATA(subj_id=configs_dict['subj_id'])
    nsd_data.make_pairs()
    nsd_data.blip2_process()
    nsd_data.blipdiffusion_process()