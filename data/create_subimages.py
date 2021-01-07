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
    hend = int(hstart + h)
    w = np.max(np.unique(trg.sum(axis=0)))
    wstart = np.argmax(trg.sum(axis=1))
    wend = int(wstart + w)
    return (wstart, wend, hstart, hend)

if __name__ == '__main__':
    imgs = np.load('imgs/imgs.npy')
    trgs = np.load('imgs/trgs.npy')
    wally_sub_imgs = []
    wally_sub_trgs = []
    for img, trg in zip(imgs, trgs):
        box = find_box(trg)
        wally_sub_imgs.append(extract_224_sub_image(img, box))
        wally_sub_trgs.append(extract_224_sub_image(trg, box))
    np.save('imgs/wally_sub_imgs.npy', np.array(wally_sub_imgs))
    np.save('imgs/wally_sub_trgs.npy', np.array(wally_sub_trgs))