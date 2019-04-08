import numpy as np
from h5py import File
import os
import torch
from torch.utils.data import Dataset

from utilities.bounding_boxes import BoundingBox


class HdfDataset(Dataset):
    def __init__(self, dataset_folder, transform=None):
        self.dataset_folder = dataset_folder
        self.data_files = os.listdir(dataset_folder)
        self.transform = transform

    def __getitem__(self, item):
        _file = os.path.join(self.dataset_folder, self.data_files[item])
        with File(_file) as f:
            pcl_data = np.array(f["point_cloud"])
            bounding_box_data = np.array(f["bounding_boxes"])
        feature_vector = self.compute_feature_vector(pcl_data)
        label_vector = self.compute_label_vector(pcl_data, bounding_box_data)
        if self.transform:
            feature_vector, label_vector = self.transform(feature_vector, label_vector)
        return feature_vector, label_vector

    def __len__(self):
        return len(self.data_files)

    def compute_feature_vector(self, pcl):
        pcl = np.rollaxis(pcl, 2, 0)  # put x,y,z dimension as the channel dimension
        feature_vector = torch.from_numpy(pcl)
        feature_vector = feature_vector.float()
        feature_vector.unsqueeze_(0)
        return feature_vector

    def compute_label_vector(self, pcl, bbx):
        # WARNING: this fails for overlapping bounding boxes - subtraction is applied multiple times in these cases
        ''' The goal of the NN should be to for each point predict the direction from the related object bounding box
        middle point.
        '''
        pcl_label = pcl.copy()
        shape = pcl.shape
        pcl_label = pcl_label.reshape(-1, 3)
        not_valid_ray_mask = np.all(pcl_label == 0, axis=1)
        changed = np.zeros(len(pcl_label))
        for bb in bbx:
            bb = BoundingBox.from_numpy(bb)
            mask = bb.check_points_inside(pcl_label)
            pcl_label -= bb.pos
            pcl_label[mask] -= bb.pos
            changed += mask
        pcl_label[not_valid_ray_mask] = 0
        nr_of_points_changed_multiple_times = np.sum(changed > 1)
        if nr_of_points_changed_multiple_times > 0:
            print("{} points were found in more than one bounding box.".format(nr_of_points_changed_multiple_times))
        pcl_label = pcl_label.reshape(*shape)
        pcl_label = np.rollaxis(pcl_label, 2, 0)  # put x,y,z dimension as the channel dimension
        label_vector = torch.from_numpy(pcl_label)
        label_vector = label_vector.float()
        label_vector.unsqueeze_(0)
        return label_vector

    def get_data_raw(self, item):
        _file = os.path.join(self.dataset_folder, self.data_files[item])
        with File(_file) as f:
            pcl_data = np.array(f["point_cloud"])
            bounding_box_data = np.array(f["bounding_boxes"])
        bounding_boxes = []
        for bbx in bounding_box_data:
            bounding_boxes.append(BoundingBox.from_numpy(bbx))
        return pcl_data, bounding_boxes
