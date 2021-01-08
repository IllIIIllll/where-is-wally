# © 2021 지성. all rights reserved.
# <llllllllll@kakao.com>
# MIT License

from utils.params import *

import numpy as np
from PIL import Image

def img_resize(img):
    h, w, _ = img.shape
    nvpanels = h / 224
    nhpanels = w / 224
    new_h, new_w = h, w
    if nvpanels * 224 != h:
        new_h = (nvpanels + 1) * 224
    if nhpanels * 224 != w:
        new_w = (nhpanels + 1) * 224
    if new_h == h and new_w == w:
        return img
    else:
        return (np.array(Image.fromarray(img).resize((new_h, new_w))) \
                / 255. - mu) /std