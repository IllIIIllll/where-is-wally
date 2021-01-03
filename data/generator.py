# © 2021 지성. all rights reserved.
# <llllllllll@kakao.com>
# MIT License

import threading

import numpy as np

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