"""
load path of large_files_for_BraVE
"""

import os
path_large_files_for_BraVE = os.path.join('..', 'large_files_for_BraVE')
if not os.path.exists(path_large_files_for_BraVE):
    os.makedirs(path_large_files_for_BraVE)