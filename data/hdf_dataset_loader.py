import numpy as np
from h5py import File
import os


class HdfLoader:
    def __init__(self, dataset_folder, shuffle=False):
        self.dataset_folder = dataset_folder
        self.data_files = os.listdir(dataset_folder)
        self.shuffle = shuffle

    def __iter__(self):
        self._order = np.arange(len(self.data_files))
        if self.shuffle:
            np.random.shuffle(self._order)
        self.iteration = 0
        return self

    def __next__(self):
        try:
            _file = os.path.join(self.dataset_folder, self.data_files[self._order[self.iteration]])
            with File(_file) as f:
                pcl_data = np.array(f["point_cloud"])
                bounding_box_data = np.array(f["bounding_boxes"])
        except IndexError:
            raise StopIteration
        self.iteration += 1
        return pcl_data, bounding_box_data
