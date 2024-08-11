__all__ = ['configs_dict']

configs_dict = {
    'subj_id' : 1,
    'dataset_name' : 'NSD', # NSD fMRI_Shape

    'NSD_ROIs' : {
        'derived_type' : 'surface',
        'roi_name' : 'prf-visualrois',
        'thresholds' : [],
    },

    'train_decoder' : {
        'batch_size' : 32, 
        'learning_rate' : 1e-4,
        'epochs' : 10,
    },

    'blip_diffusion' :{
        'negative_prompt' : 'over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate',
        'guidance_scale' : 7.5,
    },

}

blip_diffusion_coco_embedding_priori = { # COCO, only in the train set
    'min_value' : -28.20389175415039,
    'max_value' : 32.661075592041016,
    'popt' : {
        'amplitude' : 8.78480023e+08,
        'mean' : -4.75681724e-02,
        'standard_deviation' : -5.23968249e-01
    }
}  