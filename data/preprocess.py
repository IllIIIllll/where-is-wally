# © 2021 지성. all rights reserved.
# <llllllllll@kakao.com>
# MIT License

import os
import glob

import numpy as np
from PIL import Image

from utils.params import *

def get_img_num(img_file):
    return os.path.basename(img_file).split('.')[0]

def load_image(img, size=None):
    if size:
        return np.array(Image.open(img).resize(size, Image.NEAREST))
    else:
        return np.array(Image.open(img))

def load_target(img, size=None):
    return np.array(Image.open(img).convert('L').resize(size, Image.NEAREST))

if __name__ == '__main__':
    trg_files = sorted(glob.glob(f'{TRG_PATH}*.png'), key=get_img_num)
    img_files = sorted(glob.glob(f'{IMG_PATH}*.jpg'), key=get_img_num)

    size = (2800, 1760)

    imgs = np.stack([load_image(img_file, size) for img_file in img_files])
    trgs = np.stack([load_target(trg_file, size) for trg_file in trg_files])

    imgs = imgs / 255.
    trgs = trgs / 255
    imgs -= mu
    imgs /= std

    np.save('imgs/imgs.npy', imgs)
    np.save('imgs/trgs.npy', trgs)