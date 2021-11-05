#! /usr/bin/env python
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import sys

from ..utils.text_style import text_style

class dataset_TPC2d(Dataset):
    """
    TPC 2d Data 
    """
    def __init__(self, split_path, framedata_path, section_along=2, batch_size=1, shuffle=True, maximum=None):
        super(dataset_TPC2d, self).__init__()
        ts = text_style()
        # Parameter setting and validity check
        split_path = Path(split_path)
        assert split_path.exists(), \
            f"{ts.ERROR}The input split path, {split_path}, does not exists!{ts.ENDC}"
        
        for split in ['train', 'valid', 'test']:
            split_fname = split_path/f'{split}.txt'
            assert split_fname.exists(), \
                f"{ts.ERROR}Split file, {split_fname}, does not exists!{ts.ENDC}"
            
        framedata_path = Path(framedata_path)
        assert framedata_path.exists(), \
            f"{ts.ERROR}The input framedata path, {framedata_path}, does not exists!{ts.ENDC}"
        
        assert section_along in range(3), \
            f"{ts.ERROR}section_along can only be an integer in [0, 1, 2]. Frame: \n\t2: (azimuthal, z),\n\t1: (azimuthal, layer),\n\t0=(z, layer){ts.ENDC}"

        self.batch_size = batch_size
        assert isinstance(batch_size, int) and batch_size > 0, \
            f"{ts.ERROR}batch size must be a positive integer!{ts.ENDC}"
        
        self.shuffle = shuffle
        if isinstance(self.shuffle, bool):
            self.shuffle = [self.shuffle] * 3
        assert len(self.shuffle) == 3 and all([isinstance(s, bool) for s in self.shuffle]), \
            f"{ts.ERROR}shuffle must be a boolean iterable of length 3{ts.ENDC}"    
        
        assert (maximum is None) or \
            (isinstance(maximum, int) and maximum > 0) or \
            (len(maximum) == 3 and all([(isinstance(m, int) and m > 0) or (m is None) for m in maximum])), \
            f"{ts.ERROR}possible choices for maximum is None, \
                a positive integer or 3 values that are either a positive integer or a None{ts.ENDC}" 
        if isinstance(maximum, int) or (maximum is None):
            self.maximum = [maximum] * 3
        else:
            self.maximum = maximum

        # load txt file contains the symbolic links to the npy files
        section_along += 2 # The first two dimensions are sample and channel, so add 2
        self.data_loaders = {}
        for split, m, s in zip(['train', 'valid', 'test'], self.maximum, self.shuffle):
            with open(split_path/f'{split}.txt', 'r') as fp:
                file_list = fp.read().splitlines()
            file_list = [framedata_path/fname for fname in file_list]
            datum = np.array(list(map(self.__load_file, file_list)))
            # 2d sectioning
            datum = np.moveaxis(datum, section_along, 1)
            reshape_dim = datum.shape[0] * datum.shape[1]
            datum = datum.reshape(reshape_dim, *datum.shape[2:])
            np.random.shuffle(datum)

            if m is not None:
                datum = datum[: m]
            
            # apply Data loader
            loader = DataLoader(datum, batch_size=self.batch_size, shuffle=s)
            self.data_loaders[split] = loader

    def __load_file(self, fname):
        datum = np.expand_dims(np.float32(np.load(fname)), 0)
        return datum
    
    def get_split(self, split):
        return self.data_loaders[split]

    def get_splits(self):
        return self.data_loaders['train'], self.data_loaders['valid'], self.data_loaders['test']


class dataset_TPC3d(Dataset):
    """
    TPC 3d Data 
    """
    def __init__(self, split_path, framedata_path, batch_size=1, shuffle=True, maximum=None):
        super(dataset_TPC3d, self).__init__()
        # Parameter setting and validity check
        split_path = Path(split_path)
        ts = text_style()
        assert split_path.exists(), \
            f"{ts.ERROR}The input split path, {split_path}, does not exists!{ts.ENDC}"
        
        for split in ['train', 'valid', 'test']:
            split_fname = split_path/f'{split}.txt'
            assert split_fname.exists(), \
                f"{ts.ERROR}Split file, {split_fname}, does not exists!{ts.ENDC}"
            
        framedata_path = Path(framedata_path)
        assert framedata_path.exists(), \
            f"{ts.ERROR}The input framedata path, {framedata_path}, does not exists!{ts.ENDC}"

        self.batch_size = batch_size
        assert isinstance(batch_size, int) and batch_size > 0, \
            f"{ts.ERROR}batch size must be a positive integer!{ts.ENDC}"
        
        self.shuffle = shuffle
        if isinstance(self.shuffle, bool):
            self.shuffle = [self.shuffle] * 3
        assert len(self.shuffle) == 3 and all([isinstance(s, bool) for s in self.shuffle]), \
            f"{ts.ERROR}shuffle must be a boolean iterable of length 3{ts.ENDC}"    
        
        
        assert (maximum is None) or \
            (isinstance(maximum, int) and maximum > 0) or \
            (len(maximum) == 3 and all([(isinstance(m, int) and m > 0) or (m is None) for m in maximum])), \
            f"{ts.ERROR}possible choices for maximum is None, \
                a positive integer or 3 values that are either a positive integer or a None{ts.ENDC}" 
        if isinstance(maximum, int) or (maximum is None):
            self.maximum = [maximum] * 3
        else:
            self.maximum = maximum

        # load txt file contains the symbolic links to the npy files
        self.data_loaders = {}
        for split, m, s in zip(['train', 'valid', 'test'], self.maximum, self.shuffle):
            with open(split_path/f'{split}.txt', 'r') as fp:
                file_list = fp.read().splitlines()
            file_list = [framedata_path/fname for fname in file_list]
            datum = list(map(self.__load_file, file_list))
            if m is not None:
                datum = datum[: m]
            loader = DataLoader(datum, batch_size=self.batch_size, shuffle=s)
            self.data_loaders[split] = loader

    def __load_file(self, fname):
        datum = np.expand_dims(np.float32(np.load(fname)), 0)
        return datum
    
    def get_split(self, split):
        return self.data_loaders[split]

    def get_splits(self):
        return self.data_loaders['train'], self.data_loaders['valid'], self.data_loaders['test']
