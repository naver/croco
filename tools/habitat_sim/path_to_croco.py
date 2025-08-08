# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# CroCo submodule import
# --------------------------------------------------------

import sys
import os.path as path
HERE_PATH = path.normpath(path.dirname(__file__))
CROCO_REPO_PATH = path.normpath(path.join(HERE_PATH, '../../'))
# check the presence of the croco directory to make sure this is not the installed version
if path.isdir(path.join(CROCO_REPO_PATH, 'croco')):
    # workaround for sibling import
    sys.path.insert(0, CROCO_REPO_PATH)