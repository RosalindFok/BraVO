from tqdm import tqdm
from collections import Counter  

hyphen_string = '-'*16
concat_string_with_hyphen = lambda x: print(''.join([hyphen_string, x, hyphen_string]))

def analyze_volume(masking_result : dict[str, dict[str : dict[str : dict[str : float]]]]
                   ) -> dict[str : dict[str : int]]:
    concat_string_with_hyphen(analyze_volume.__name__)
    volume_derived_result = {'MSE':{'MTL':0,'thalamus':0}, 'MAE':{'MTL':0,'thalamus':0}}
    for index, rois_and_category in tqdm(masking_result.items(), desc=f'Analyzing volume', leave=True):
        for derived_type_or_general_category, values in rois_and_category.items():
            if derived_type_or_general_category == 'volume':
                MTL_mse = values['MTL']['whole']['MSE']
                MTL_mae = values['MTL']['whole']['MAE']
                thalamus_mse = values['thalamus']['whole']['MSE']
                thalamus_mae = values['thalamus']['whole']['MAE']
                # MSE
                if MTL_mse < thalamus_mse:
                    volume_derived_result['MSE']['MTL'] += 1
                else: 
                    volume_derived_result['MSE']['thalamus'] += 1
                # MAE
                if MTL_mae < thalamus_mae:
                    volume_derived_result['MAE']['MTL'] += 1
                else: 
                    volume_derived_result['MAE']['thalamus'] += 1
    concat_string_with_hyphen('End')
    return volume_derived_result


def analyze_volume_MTL(masking_result : dict[str, dict[str : dict[str : dict[str : float]]]]
                       ) -> dict[int : dict[str : any]]:
    concat_string_with_hyphen(analyze_volume_MTL.__name__)
    mtl_result = {} # {key=index, value=dict}
    for index, rois_and_category in tqdm(masking_result.items(), desc=f'Analyzing volume MTL', leave=True):
        index = int(index)
        mtl_result[index] = {} # {key=roi_name, value=dict}
        for derived_type_or_general_category, values in rois_and_category.items():
            # region of ROIs
            if derived_type_or_general_category == 'volume':
                for rois_name, regions in values.items():
                    if not rois_name == 'MTL':
                        continue
                    min_mse_region, min_mae_region = '', ''
                    min_mse, min_mae = 1e9, 1e9
                    for region_name, metrics in regions.items():
                        if region_name == 'whole':
                            continue
                        else:
                            if metrics['MSE'] < min_mse:
                                min_mse = metrics['MSE']
                                min_mse_region = region_name
                            if metrics['MAE'] < min_mae:
                                min_mae = metrics['MAE']
                                min_mae_region = region_name
                    mtl_result[index] = {'MSE':min_mse_region, 'MAE':min_mae_region}
            # category
            elif derived_type_or_general_category not in ['volume', 'surface']: 
                mtl_result[index]['supercategory'] = derived_type_or_general_category
                mtl_result[index]['category'] = values
    
    # the relationship between category and region of MTL
    mse_region_list, mae_region_list, supercategory_list = [], [], []
    for index, regions_category in mtl_result.items():
        for key, value in regions_category.items():
            if key == 'MSE':
                mse_region_list.append(value)
            elif key == 'MAE':
                mae_region_list.append(value)
            elif key == 'supercategory':
                supercategory_list.append(value)
    assert len(mse_region_list) == len(mae_region_list) == len(supercategory_list), 'The length of lists is not equal'
    # count the frequency of each element in the lists
    element_count = Counter(mse_region_list)
    element_count = Counter(mae_region_list)

    concat_string_with_hyphen('End')
    return mtl_result

def analyze_volume_thalamus(masking_result : dict[str, dict[str : dict[str : dict[str : float]]]]
                            ) -> dict[str : dict[str : float]]:
    concat_string_with_hyphen(analyze_volume_thalamus.__name__)
    thalamus_result = {} # {key=index, value=dict}
    for index, rois_and_category in tqdm(masking_result.items(), desc=f'Analyzing volume thalamus', leave=True):
        for derived_type_or_general_category, values in rois_and_category.items():
            # region of ROIs
            if derived_type_or_general_category == 'volume':
                for rois_name, regions in values.items():
                    if not rois_name == 'thalamus':
                        continue
                    for region_name, metrics in regions.items():
                        if region_name == 'whole':
                            continue
                        if region_name in thalamus_result:
                            thalamus_result[region_name]['MSE'] += metrics['MSE']
                            thalamus_result[region_name]['MAE'] += metrics['MAE']
                        else:
                            thalamus_result[region_name] = {'MSE':metrics['MSE'], 'MAE':metrics['MAE']}
    print('\t\tMSE\tMAE')
    for region_name, metrics in thalamus_result.items():
        tab = '\t\t' if len(region_name) < 8 else '\t'
        print(f"{region_name}{tab}{(metrics['MSE'])/len(masking_result):.4f}\t{(metrics['MAE'])/len(masking_result):.4f}")  
    concat_string_with_hyphen('End')
    return thalamus_result

def analyze_surface_floc(masking_result : dict[str, dict[str : dict[str : dict[str : float]]]]
                         ) -> None:
    concat_string_with_hyphen(analyze_surface_floc.__name__)
    supercategory_list, mse_rois_list, mae_rois_list = [], [], []
    for index, rois_and_category in tqdm(masking_result.items(), desc=f'Analyzing surface floc', leave=True):
        for derived_type_or_general_category, values in rois_and_category.items():
            # region of ROIs
            if derived_type_or_general_category == 'surface':
                mse_roi, mae_roi = '', ''
                min_mse, min_mae = 1e9, 1e9
                for rois_name, regions in values.items():
                    if not 'floc-' in rois_name:
                        continue
                    else:
                        for region_name, metrics in regions.items():
                            if region_name == 'whole':
                                mse, mae = metrics['MSE'], metrics['MAE']
                                if mse < min_mse:
                                    min_mse = mse
                                    mse_roi = rois_name
                                if mae < min_mae:
                                    min_mae = mae
                                    mae_roi = rois_name
                mse_rois_list.append(mse_roi)
                mae_rois_list.append(mae_roi)
            # category
            elif derived_type_or_general_category not in ['volume', 'surface']: 
                supercategory_list.append(derived_type_or_general_category)
    
    # 
    for x, y, z in zip(supercategory_list, mse_rois_list, mae_rois_list):
        print(f"{x}\t{y}\t{z}")

    concat_string_with_hyphen('End')