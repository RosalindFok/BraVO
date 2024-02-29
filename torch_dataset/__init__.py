import os
import yaml
import warnings

"""
    torch_dataset模块
    使用PyTorch框架提供的Dataset类来制作自己的数据集
    注: 此模块中所用路径以main.py文件所在位置为工作路径
"""

def read_config(config_path : str) -> dict:
    '''
    读取yaml配置文件
    '''
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            return config
    except Exception as e:
            print(f"读取配置文件 {config_path} 时发生错误: {e}")
            return None

def check_path_exists(*paths : str) -> None:
    '''
    检查每个path所指向的文件夹/文件是否存在 
    '''
    for path in paths:
        if not os.path.exists(path):
            warnings.simplefilter(f'{path} dose not exist!') 

def check_and_make_folder(*paths : str) -> bool:
    '''
    检查每个文件夹是否存在 若不存在则创建 不输出任何警告
    '''
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
    return True


config_path = os.path.join('.','config.yaml')
config_info = read_config(config_path=config_path)

hdf5_dir = os.path.join('..', 'hdf5_files')
check_and_make_folder(hdf5_dir)

if not config_info == None:
    dataset_path = os.path.join(config_info['path']['parent'], config_info['path']['dataset'])
    algnauts_path = os.path.join(dataset_path, config_info['path']['algnauts'])
    nsd_path = os.path.join(dataset_path, config_info['path']['nsd'])
    nod_path = os.path.join(dataset_path, config_info['path']['nod'])
    check_path_exists(algnauts_path, nsd_path, nod_path)
else:
     print(f"配置文件不存在, 请检查!")
     exit(1)