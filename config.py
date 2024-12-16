__all__ = ['configs_dict']

configs_dict = {
    'subj_id' : 1,
    'dataset_name' : 'NSD', # NSD fMRI_Shape

    'functional_space' : 'func1mm', # func1mm func1pt8mm
    'embedding_space'  : 'blip2',   # blip2 blipdiffusion

    'NSD_ROIs' : {
        'derived_type' : 'surface',
        'roi_name' : 'streams',
        'thresholds' : {
            'primary_visual_cortex' : [1], # early    1 
            'higher_visual_cortex'  : [5], # ventral  5
            'whole_cortex'          : [] , # 1~7
        }
    },

    'train_decoder' : {
        'batch_size' : 8, 
        'learning_rate' : 1e-5,
        'epochs' : 20,
    },

    'blip_diffusion' : {
        'negative_prompt' : 'over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate',
        'iter_seed' : 6,
        'guidance_scale' : 8.5,
        'num_inference_steps' : 500, 
    },

    'blip2' : {
        'max_length' : 50,
        'min_length' : 20,
    },
}
