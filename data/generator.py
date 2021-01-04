# © 2021 지성. all rights reserved.
# <llllllllll@kakao.com>
# MIT License

import threading
import random

import numpy as np
from tensorflow.keras.utils import to_categorical

class BatchIndices(object):
    def __init__(self, n, bs, shuffle=False):
        self.n, self.bs, self.shuffle = n, bs, shuffle
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.idxs = (
            np.random.permutation(self.n)
            if self.shuffle else np.arange(0, self.n)
        )
        self.curr = 0

    def __next__(self):
        with self.lock:
            if self.curr >= self.n:
                self.reset()
            ni = min(self.bs, self.n - self.curr)
            res = self.idxs[self.curr:self.curr + ni]
            self.curr += ni
            return res

class segm_generator(object):
    def __init__(self, x, y, bs=64, out_sz=(224,224), train=True, wally=True):
        self.x, self.y, self.bs, self.train = x,y,bs,train
        self.wally = wally
        self.n = x.shape[0]
        self.ri, self.ci = [], []
        for i in range(self.n):
            ri, ci, _ = x[i].shape
            self.ri.append(ri), self.ci.append(ci)
        self.idx_gen = BatchIndices(self.n, bs, train)
        self.ro, self.co = out_sz
        self.ych = self.y.shape[-1] if len(y.shape)==4 else 1

    def get_slice(self, i,o):
        start = random.randint(0, i-o) if self.train else (i-o)
        return slice(start, start+o)

    def get_item(self, idx):
        slice_r = self.get_slice(self.ri[idx], self.ro)
        slice_c = self.get_slice(self.ci[idx], self.co)
        x = self.x[idx][slice_r, slice_c]
        y = self.y[idx][slice_r, slice_c]
        if self.train and (random.random()>0.5):
            y = y[:,::-1]
            x = x[:,::-1]
        if not self.wally and np.sum(y)!=0:
            return None
        return x, to_categorical(y, num_classes=2)

    def __next__(self):
        idxs = self.idx_gen.__next__()
        items = []
        for idx in idxs:
            item = self.get_item(idx)
            if item is not None:
                items.append(item)
        if not items:
            return None
        xs,ys = zip(*tuple(items))
        return np.stack(xs), np.stack(ys)