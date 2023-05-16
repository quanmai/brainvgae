import torch
import os
import numpy as np
from torch_geometric.data import InMemoryDataset
from .read_abide_stats_parall import read_data
from collections.abc import Mapping


# I will divide dataset into training+val+test set


class ABIDEDataset(InMemoryDataset):
    def __init__(self, root, name, augment=True, transform=None, pre_transform=None):
        # super().__init__()
        self.root = root
        self.name = name
        self.augment = augment
        super(ABIDEDataset, self).__init__(root,transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        data_dir = os.path.join(self.root,'raw')
        onlyfiles = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
        onlyfiles.sort()
        return onlyfiles

    @property
    def processed_file_names(self):
        return 'data.pt' #.format(self.numnode)


    def process(self):
        # Read data into huge `Data` list.
        self.data, self.slices = read_data(self.raw_dir, self.augment)
        # print(self.data.)
        
        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])
        

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))

def nested_iter(mapping):
    for key, value in mapping.items():
        if isinstance(value, Mapping):
            for inner_key, inner_value in nested_iter(value):
                yield inner_key, inner_value
        else:
            yield key, value