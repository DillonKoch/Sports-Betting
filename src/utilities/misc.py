# ==============================================================================
# File: misc.py
# Project: allison
# File Created: Tuesday, 28th February 2023 2:12:57 pm
# Author: Dillon Koch
# -----
# Last Modified: Tuesday, 28th February 2023 2:12:59 pm
# Modified By: Dillon Koch
# -----
#
# -----
# miscellaneous utility files
# ==============================================================================

import json


def leagues():
    return ['NFL', 'NBA', 'NCAAF', 'NCAAB']


def load_json(path):
    with open(path, 'r') as f:
        config = json.load(f)
    return config
