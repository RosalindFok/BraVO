__all__ = ['configs_dict']

configs_dict = {
    'subj_id' : 1,
    'dataset_name' : 'NSD', # NSD fMRI_Shape

    'NSD_ROIs' : {
        'derived_type' : 'surface',
        'roi_name' : 'corticalsulc',
        'thresholds' : [],
    },

    'train_decoder' : {
        'batch_size' : 32, # 32 <--> 1 cuda 
        'learning_rate' : 1e-4,
        'epochs' : 130,
    },

    'blip_diffusion' :{
        'negative_prompt' : 'over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate',
        'iter_seed' : 8888,
        'guidance_scale' : 7.5,
        'num_inference_steps' : 500, 
    },

}
