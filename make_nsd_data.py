import os
import cv2
import h5py
import time
import json
import scipy.io
import numpy as np
import pandas as pd
from tqdm import tqdm

from models import BLIP2_Tools
from utils import NSD_dir_path, BraVO_saved_dir_path
from utils import join_paths, read_nii_file, save_nii_file, check_and_make_dirs, read_json_file, merge_dicts_if_no_conflict

class NSD_DATA():
    def __init__(self, NSD_dir_path : str = NSD_dir_path, subj_id : int | str = None) -> None:
        super().__init__()
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
    
    def read_coco_annotation(self) -> dict[int, list[str]]:
        def __extract_captions__(captions_annotations : list[dict[str, any]]) -> dict[int, list[str]]:
            annotations = {} # {key=id : value=[caption1, caption2, ...]}
            # some pictures have multiple captions
            for ca in captions_annotations:
                if not ca['image_id'] in annotations:
                    annotations[ca['image_id']] = [ca['caption']]
                else:
                    annotations[ca['image_id']].append(ca['caption'])
            return annotations
        
        captions_train2017 = read_json_file(path=join_paths(self.coco_annotation_dir_path, 'captions_train2017.json'))
        captions_val2017 = read_json_file(path=join_paths(self.coco_annotation_dir_path, 'captions_val2017.json'))
        captions_train_annotations = captions_train2017['annotations']
        captions_val_annotations = captions_val2017['annotations']
        train_annotations = __extract_captions__(captions_train_annotations)
        val_annotations = __extract_captions__(captions_val_annotations)
        # captions_dict is {key=id : value=[caption1, caption2, ...]}
        captions_dict = merge_dicts_if_no_conflict(train_annotations, val_annotations) 
        
        return captions_dict
    
    def make_pairs(self) -> None:
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
        captions_dict = self.read_coco_annotation()
        
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
                        # image: opencv writes via BGR, BLIP-2 encodes via RGB
                        image = cv2.cvtColor(imgBrick[KID_73], cv2.COLOR_RGB2BGR)
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        cv2.imwrite(join_paths(saved_path, 'image.png'), image)
                        # caption and other infos
                        OLD_flag = int(trial[column_of_ISOLD]) # 0 was novel, 1 was old
                        captions_list = captions_dict[stim_info[KID_73]] # list[str]

                        # Encode image and caption via BLIP-2, get embedding whose dim=768
                        save_info_json = {'ISOLD':'novel' if OLD_flag == 0 else 'old', 'caption':{}}
                        image_embedding = BLIP2_Tools.blip2_encoder(mode='i', image_rgb=image_rgb)
                        save_embedding_npz = {'image':image_embedding, 'caption':{}, 'multimodal':{}}
                        for index, caption in enumerate(captions_list):
                            caption_embedding = BLIP2_Tools.blip2_encoder(mode='t', caption=caption)
                            multimodal_embedding = BLIP2_Tools.blip2_encoder(mode='m',image_rgb=image_rgb, caption=caption)
                            save_embedding_npz['caption'][index] = caption_embedding
                            save_embedding_npz['multimodal'][index] = multimodal_embedding
                            save_info_json['caption'][str(index)] = caption
                        # save npz, which contains embeddings  of image, caption, and multimodal. 
                        np.savez(join_paths(saved_path, 'embedding.npz'), save_embedding_npz)
                        # save json, which contains ISOLD, captions.
                        with open(join_paths(saved_path, 'info.json'), 'w', encoding='utf-8') as f:
                            json.dump(save_info_json, f, indent=4)
                            
                    # incorrect trial
                    else:
                        continue
            else:
                raise NotImplementedError(f'Space type: {space_type} is not supported.')
        print(f'{self.subj} has {len(os.listdir(train_saved_dir_path))} pairs in train set, {len(os.listdir(test_saved_dir_path))} pairs in test set.')

# make pairs of NSD
NSD_DATA(subj_id=1)        
NSD_DATA(subj_id=2)        
NSD_DATA(subj_id=5)        
NSD_DATA(subj_id=7)        