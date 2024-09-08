__all__ = ['configs_dict']

configs_dict = {
    'subj_id' : 1,
    'dataset_name' : 'NSD', # NSD fMRI_Shape

    'NSD_ROIs' : {
        'derived_type' : 'surface',
        'roi_name' : 'prf-visualrois',# corticalsulc
        # primary_visual_cortex : [1, 2, 3, 4],    v1v, v1d, v2v, v2d
        # higher_visual_cortex  : [3, 4, 5, 6, 7]  v2v, v2d, v3v, v3d, hv4
        'thresholds' : [1, 2, 3, 4],
    },

    'train_decoder' : {
        'batch_size' : 256, 
        'learning_rate' : 1e-3,
        'epochs' : 20,
    },

    'blip_diffusion' :{
        'negative_prompt' : 'over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate',
        'iter_seed' : 8888,
        'guidance_scale' : 7.5,
        'num_inference_steps' : 500, 
    },

}
