import collections

import numpy as np


Data = collections.namedtuple('Data', 's a a_i p_a r sp R done')


class Buffer(object):
    def __init__(self, val, max_size):
        self.buffer = np.zeros((max_size,) + val.shape, dtype=val.dtype)
        self.max_size = max_size

        self.len = 0
        self.position = 0

        # print(self.buffer.shape, self.buffer.dtype)

    def add(self, val):
        self.buffer[self.position] = val.copy()

        self.position = (self.position + 1) % self.max_size
        self.len += 1

    def __getitem__(self, key):
        return self.buffer[key]

    def __len__(self):
        return min(self.max_size, self.len)


class ReplayBuffer(object):
    def __init__(self, max_size):
        self.buffers = dict()
        self.max_size = max_size

    def add(self, data):
        for key in data._fields:
            val = getattr(data, key)

            if key not in self.buffers:
                self.buffers[key] = Buffer(val, self.max_size)

            self.buffers[key].add(val)

    def __getitem__(self, idx):
        result = list()

        for key in Data._fields:
            result.append(self.buffers[key][idx])

        return result

    def __len__(self):
        lens = list()

        for _, val in self.buffers.items():
            lens.append(len(val))

        assert min(lens) == max(lens)

        return lens[0]
