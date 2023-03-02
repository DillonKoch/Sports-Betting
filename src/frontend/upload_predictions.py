# ==============================================================================
# File: upload_predictions.py
# Project: allison
# File Created: Tuesday, 28th February 2023 7:22:56 am
# Author: Dillon Koch
# -----
# Last Modified: Tuesday, 28th February 2023 7:22:57 am
# Modified By: Dillon Koch
# -----
#
# -----
# uploading predictions from /data/predictions to MongoDB frontend
# ==============================================================================


from os.path import abspath, dirname
import sys

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


class Upload_Predictions:
    def __init__(self):
        pass

    def run(self):  # Run
        pass


if __name__ == '__main__':
    x = Upload_Predictions()
    self = x
    x.run()
