import torch
from argparse import ArgumentParser
import os
import numpy as np
from h5py import File

from scripts.train import PointCloudMapRegressionNet
from data.hdf_dataset_loader import HdfDataset


class InferenceDataset(HdfDataset):
    def __getitem__(self, item):
        data_file = self.data_files[item]
        _file = os.path.join(self.dataset_folder, data_file)
        with File(_file) as f:
            pcl_data = np.array(f["point_cloud"])
        feature_vector = self.compute_feature_vector(pcl_data)
        return feature_vector, data_file, pcl_data


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("data_folder")
    parser.add_argument("save_folder")
    args = parser.parse_args()

    net = PointCloudMapRegressionNet()
    net.load_state_dict(torch.load(args.model))

    files = os.listdir(args.data_folder)

    data_loader = InferenceDataset(args.data_folder)

    for i in range(len(data_loader)):
        feature_vector, file_name, pcl = data_loader[i]
        output = net(feature_vector)
        output_file_name = os.path.join(args.save_folder, "out_" + file_name)
        with File(output_file_name, "w") as f:
            f.create_dataset("point_cloud", data=pcl)
            f.create_dataset("object_vectors", data=output)
