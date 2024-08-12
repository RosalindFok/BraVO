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
