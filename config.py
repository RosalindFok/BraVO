__all__ = ['configs_dict']

configs_dict = {
    'subj_id' : 1,
    'dataset_name' : 'NSD', # NSD fMRI_Shape

    'NSD_ROIs' : {
        'derived_type' : 'surface',
        'roi_name' : 'prf-visualrois',# corticalsulc
        'thresholds' : {
            'primary_visual_cortex' : [1, 2],    # v1v, v1d
            'higher_visual_cortex'  : [3, 4, 5, 6, 7]  # v2v, v2d, v3v, v3d, hv4
        }
    },

    'train_decoder' : {
        'batch_size' : 128, 
        'learning_rate' : 5e-4,
        'epochs' : 100,
    },

    'blip_diffusion' :{
        'negative_prompt' : 'over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate',
        'iter_seed' : 8888,
        'guidance_scale' : 7.5,
        'num_inference_steps' : 500, 
    },

    'blip_caption' : {
        'prompt' : 'Please provide a detailed description of this image, including all visible elements such as objects, people, settings, actions, colors, and emotions, and please do not generate repetitive statements.'
    }
}
