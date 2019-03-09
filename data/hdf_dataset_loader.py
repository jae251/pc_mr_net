import numpy as np
from h5py import File
import os
from itertools import product
import torch

from utilities.bounding_boxes import BoundingBox


class HdfLoader:
    def __init__(self, dataset_folder, shuffle=False, feature_distances=(.2, .4, .8)):
        self.dataset_folder = dataset_folder
        self.data_files = os.listdir(dataset_folder)
        self.shuffle = shuffle
        self.feature_distances = feature_distances

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
        feature_vector = self.compute_feature_vector(pcl_data)
        label_vector = self.compute_label_vector(pcl_data, bounding_box_data)
        return feature_vector, label_vector

    def __len__(self):
        return len(self.data_files)

    # def compute_feature_vector(self, pcl):
    #     ''' Every point gets assigned a set of features for each entry in self.feature_distances.
    #     The sphere around each point defined by each feature distance gets subdivided into 8 compartments
    #     and can be thought of as a directional property.
    #     The points inside the regions are counted. For each point, each feature distance and each direction the point
    #     count is entered into the feature vector.
    #     This set of features can be thought of as approximated directional point density.
    #     '''
    #     local_pcl = pcl[np.newaxis] - pcl[:, np.newaxis]
    #     distance_matrix = np.linalg.norm(local_pcl, axis=2)
    #     feature_vector = np.zeros((len(pcl), 8 * len(self.feature_distances)))
    #     not_zero_mask = distance_matrix != 0
    #     for i, dist in enumerate(self.feature_distances):
    #         dist_mask = (distance_matrix <= dist) * not_zero_mask
    #         for n, (k1, k2, k3) in enumerate(product((1, -1), repeat=3)):  # iterate over the 8 space segments
    #             located_in_segment = (local_pcl[:, :, 0] * k1 >= 0) * \
    #                                  (local_pcl[:, :, 1] * k2 >= 0) * \
    #                                  (local_pcl[:, :, 2] * k3 >= 0)
    #             point_count_in_segment = np.sum(located_in_segment * dist_mask, axis=1)
    #             feature_vector[:, i * 8 + n] = point_count_in_segment
    #     return torch.from_numpy(feature_vector)

    def compute_feature_vector(self, pcl):
        feature_vector = torch.from_numpy(pcl)

    def compute_label_vector(self, pcl, bbx):
        # WARNING: this fails for overlapping bounding boxes - subtraction is applied multiple times in these cases
        ''' The goal of the NN should be to for each point predict the direction from the related object bounding box
        middle point.
        '''
        pcl_label = pcl.copy()
        changed = np.zeros(len(pcl_label))
        for bb in bbx:
            bb = BoundingBox.from_numpy(bb)
            mask = bb.check_points_inside(pcl)
            pcl_label[mask] -= bb.pos
            changed += mask
        nr_of_points_changed_multiple_times = np.sum(changed > 1)
        if nr_of_points_changed_multiple_times > 0:
            print("{} points were found in more than one bounding box.".format(nr_of_points_changed_multiple_times))
        return torch.from_numpy(pcl_label)
