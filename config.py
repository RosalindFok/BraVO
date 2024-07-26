__all__ = ['configs_dict']

configs_dict = {
    'subj_id' : 1,
    'batch_size' : 10, 
    'learning_rate' : 1e-4,
    'derived_type' : 'surface', # surface, volume
    # surface-(corticalsulc, floc-bodies, floc-faces, floc-places, floc-words, 
    #          HCP_MMP1, Kastner2015, nsdgeneral, prf-eccrois, prf-visualrois, streams)
    # volume-(MTL, thalamus)
    'roi_name' : 'corticalsulc', 
    'threshold' : 0,
    'negative_prompt' : 'over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate',
    'guidance_scale' : 7.5,
}