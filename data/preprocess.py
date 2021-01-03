# © 2021 지성. all rights reserved.
# <llllllllll@kakao.com>
# MIT License

import numpy as np
from PIL import Image

def load_image(img, size=None):
    if size:
        return np.array(Image.open(img).resize(size, Image.NEAREST))
    else:
        return np.array(Image.open(img))