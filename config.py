__all__ = ['configs_dict']

configs_dict = {
    'subj_id' : 1,

    'train_encoder' : {
        'batch_size' : 200, 
        'learning_rate' : 1e-4,
        'epochs' : 10,
    },

    'blip_diffusion' :{
        'negative_prompt' : 'over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate',
        'guidance_scale' : 7.5,
    },


}