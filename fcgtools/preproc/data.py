import json
import struct
import numpy as np
from pathlib import Path

def make_view(filename, dtype):
    mmap = np.memmap(filename, mode = 'r', dtype = dtype)
    view = memoryview(mmap)
    return view


class DataFiles:

    def __init__(self, path):
        self.js = path + '.json'
        self.pos = path + '.pos'
        self.len = path + '.len'
        self.dat = path + '.dat'


class MemmapDataWriter:

    def __init__(self, path):
        self.files = DataFiles(str(path))
        self.i = 0
        Path(path).parent.mkdir(parents = True, exist_ok = True)
        self.json_file = open(self.files.js, 'w')
        self.pos_file = open(self.files.pos, 'wb')
        self.len_file = open(self.files.len, 'wb')
        self.dat_file = open(self.files.dat, 'wb')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        json.dump(self.i, self.json_file)
        self.json_file.close()
        self.pos_file.close()
        self.len_file.close()
        self.dat_file.close()

    def write(self, lst):
        self.i += 1
        point = self.dat_file.tell()
        self.pos_file.write(struct.pack('q', point))
        self.len_file.write(struct.pack('h', len(lst)))

        for x in lst:
            self.dat_file.write(struct.pack('i', x))


class MemmapData:

    def __init__(self, path):
        self.files = DataFiles(str(path))

        with open(self.files.js, 'r') as f:
            self.i = json.load(f)

        self.pos_view = make_view(self.files.pos, np.int64)
        self.len_view = make_view(self.files.len, np.int16)
        self.dat_view = make_view(self.files.dat, np.int32)

    def __len__(self):
        return self.i

    def __getitem__(self, index):
        point = self.pos_view[index]
        size = self.len_view[index]
        x = np.frombuffer(self.dat_view, dtype = np.int32, count = size, offset = point)
        return x.tolist()

