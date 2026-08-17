import numpy as np
import cv2
from PIL import Image
import argparse
from glob import glob
import os

# Prepare your data to kitti format.
class GenTranspGT():
    def __init__(self, args):
        super(GenTranspGT, self).__init__()
        self.filelist = glob(os.path.join(args.data_path, "**/*.npy"), recursive=True)

        for file in self.filelist:
            print(file)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate GT for Transparent")
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()
    GenTranspGT(args)
