"""
load path of large_files_for_BraVE
"""

import os
path_large_files_for_BraVE = os.path.join('..', 'large_files_for_BraVE')
if not os.path.exists(path_large_files_for_BraVE):
    os.makedirs(path_large_files_for_BraVE)

#### files in large_files_for_BraVE {
# bert-base-uncased(directory)
# model_base.pth(file)
# model_base_capfilt_large.pth(file)
# model_large.pth(file)
# ViT-B-32.pt(file)
# }
####