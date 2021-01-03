# © 2021 지성. all rights reserved.
# <llllllllll@kakao.com>
# MIT License

import numpy as np

def extract_224_sub_image(img, box):
    wstart, wend, hstart, hend = box
    i = img[wend-224:wstart+224, hend-224:hstart+224]
    if not i.shape[0]:
        return img[wstart:wstart+225, hend-224:hstart+224]
    else:
        return i

def find_box(trg):
    h = np.max(np.unique(trg.sum(axis=1)))
    hstart = np.argmax(trg.sum(axis=0))
    hend = int(hstart+h)
    w = np.max(np.unique(trg.sum(axis=0)))
    wstart = np.argmax(trg.sum(axis=1))
    wend = int(wstart+w)
    return (wstart, wend, hstart, hend)