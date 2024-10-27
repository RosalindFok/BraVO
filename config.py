__all__ = ['configs_dict']

configs_dict = {
    'subj_id' : 1,
    'dataset_name' : 'NSD', # NSD fMRI_Shape

    'NSD_ROIs' : {
        'derived_type' : 'surface',
        'roi_name' : 'streams',
        'thresholds' : {
            'primary_visual_cortex' : [1], # early 
            'higher_visual_cortex'  : [5]  # ventral 
        }
    },

    'train_decoder' : {
        'batch_size' : 256, 
        'learning_rate' : 2e-4,
        'epochs' : 20,
    },

    'blip_diffusion' :{
        'negative_prompt' : 'over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate',
        'iter_seed' : 6,
        'guidance_scale' : 8.5,
        'num_inference_steps' : 500, 
    },

    'blip_caption' : {
        'prompt' : 'Please provide a detailed description of this image, including all visible elements such as objects, people, settings, actions, colors, and emotions, and please do not generate repetitive statements.'
    }
}
