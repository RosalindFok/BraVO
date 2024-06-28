import os
import cv2
import h5py
import scipy.io
import pandas as pd
from tqdm import tqdm

from utils import NSD_dir_path
from utils import join_paths, read_nii_file, sort_dict_via_keys, extract_number_from_string

class NSD_DATA():
    def __init__(self, NSD_dir_path : str = NSD_dir_path, subj_id : int | str = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # subj_id
        self.subj = ''.join(['subj0', str(subj_id)])
        
        ## nsddata
        self.nsddata_dir_path = join_paths(NSD_dir_path, 'nsddata')
        self.nsddata_ppdata_dir_path = join_paths(self.nsddata_dir_path, 'ppdata')
        # Info: https://cvnlab.slite.page/p/fRv4lz5V2F/Behavioral-data#2bdd55ef
        self.behav_responses_tsv_file_path = join_paths(self.nsddata_ppdata_dir_path, self.subj, 'behav', 'responses.tsv')
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
        self.coco_annotation_dir_path = join_paths(NSD_dir_path, 'nsddata_stimuli', 'stimuli', 'nsd', 'annotations')

    def read_behav_responses_tsv(self):
        data_frame = pd.read_csv(self.behav_responses_tsv_file_path, sep='\t', encoding='utf-8')
        print(data_frame.head(100))
        print(data_frame['73KID']) # 看hdf5里面写的索引从1开始

    def read_expdesign_mat(self):
        mat_contents = scipy.io.loadmat(self.expdesign_mat_file_path)
        print(mat_contents.keys())
    
    def read_stim_info_csv(self):
        data_fram = pd.read_csv(self.stim_info_csv_file_path)
        print(data_fram.head())

    def read_betas(self, space_type : str):
        if space_type == 'fsaverage':
            # FsAverage: https://mne.tools/dev/auto_tutorials/forward/10_background_freesurfer.html
            fsaverage_dir_path = join_paths(self.nsddata_betas_ppdata_dir_path, 'fsaverage')
            # Info: https://cvnlab.slite.page/p/6CusMRYfk0/Functional-data-NSD#cfb79694
            betas_fithrf_GLMdenoise_RR_dir_path = join_paths(fsaverage_dir_path, 'betas_fithrf_GLMdenoise_RR')
            # Info: https://cvnlab.slite.page/p/6CusMRYfk0/Functional-data-NSD#8ff511f4
            betas_session_mgh_files_paths_list = [join_paths(betas_fithrf_GLMdenoise_RR_dir_path, x) for x in os.listdir(betas_fithrf_GLMdenoise_RR_dir_path) if x.endswith('.mgh') and 'h.betas_session' in x]
            beta_session_dict = {} # {key=session_id, value={key=lh or rh, value=data in mgh file}}
            for file_path in tqdm(betas_session_mgh_files_paths_list, desc=f'Reading mgh files', leave=True):
                header, data = read_nii_file(file_path) # dims=[163842,1,1,750]; dtype=float64
                file_name = file_path.split(os.sep)[-1]
                session_id = extract_number_from_string(file_name)
                if not session_id in beta_session_dict:
                    beta_session_dict[session_id] = {}
                else:
                    beta_session_dict[session_id][file_name[:2]] = data
            sorted_beta_session_dict = sort_dict_via_keys(beta_session_dict)

        elif space_type == 'func1mm':
            # func1mm and func1pt8mm: https://cvnlab.slite.page/p/6CusMRYfk0/Functional-data-NSD#9065a649
            func1mm_dir_path = join_paths(self.nsddata_betas_ppdata_dir_path, 'func1mm')
            betas_fithrf_GLMdenoise_RR_dir_path = join_paths(func1mm_dir_path, 'betas_fithrf_GLMdenoise_RR')
            # Info: https://cvnlab.slite.page/p/6CusMRYfk0/Functional-data-NSD#3e1740b1
            betas_session_nii_files_paths_list = [join_paths(betas_fithrf_GLMdenoise_RR_dir_path, x) for x in os.listdir(betas_fithrf_GLMdenoise_RR_dir_path) if x.endswith('.nii.gz') and 'betas_session' in x]
            beta_session_dict = {} # {key=session_id, value=data in hdf5 file}
            for file_path in tqdm(betas_session_nii_files_paths_list, desc=f'Reading nii files', leave=True):
                file_name = file_path.split(os.sep)[-1]
                session_id = extract_number_from_string(file_name)
                file_path = file_path.replace('.hdf5', '.')
                header, data = read_nii_file(file_path) # dims=(145, 186, 148, 750); dtype=float64
                beta_session_dict[session_id] = data
            sorted_beta_session_dict = sort_dict_via_keys(beta_session_dict)

        else:
            raise NotImplementedError(f'Space type: {space_type} is not supported.')
        
    def read_stimuli_hdf5(self):
        with h5py.File(self.nsddata_stimuli_hdf5_file_path, 'r') as f:
            # imgBrick is 3 channels x 425 pixels x 425 pixels x 73,000 images and is in uint8 format. 
            # These images are shown on a gray background with RGB value (127,127,127).
            imgBrick = f['imgBrick']
    
    # 这里保存图片噜 图片 73KID-1 COCO中的索引 COCO中的描述性文本 fMRI
    # matrix_bgr = cv2.cvtColor(imgBrick[0], cv2.COLOR_RGB2BGR)
    # cv2.imwrite('output_image.png', matrix_bgr)

# Test
subj01 = NSD_DATA(subj_id=1)
subj01.read_behav_responses_tsv()