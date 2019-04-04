import torch
import os
import numpy as np
from h5py import File

from utilities.visualize_net_output import Visualizer
from utilities.bounding_boxes import BoundingBox
from net_def import PointCloudMapRegressionNet
from data.hdf_dataset_loader import HdfDataset


class InferenceDataset(HdfDataset):
    def __getitem__(self, item):
        data_file = self.data_files[item]
        _file = os.path.join(self.dataset_folder, data_file)
        with File(_file) as f:
            pcl_data = np.array(f["point_cloud"])
            bboxes = np.array(f["bounding_boxes"])
        bounding_boxes = []
        for bb in bboxes:
            bounding_boxes.append(BoundingBox.from_numpy(bb))
        feature_vector = self.compute_feature_vector(pcl_data)
        return feature_vector, pcl_data.reshape(-1, 3), bounding_boxes


MODEL_DIR = "net_weights"
DATA_DIR = "../../data/dataset_one_car/train"
# DATA_DIR = "../../data/dataset_one_car/eval"


if __name__ == '__main__':
    net = PointCloudMapRegressionNet()
    model = os.path.join(MODEL_DIR, sorted(os.listdir(MODEL_DIR))[-1])
    net.load_state_dict(torch.load(model, map_location="cpu"))

    data_loader = InferenceDataset(DATA_DIR)
    feature_vector, pcl_data, bounding_boxes = data_loader[np.random.randint(0, len(data_loader))]
    output = net(feature_vector).data.numpy()[0].swapaxes(0, 2).reshape(-1, 3)
    vis = Visualizer()
    mask = np.any(pcl_data != 0, axis=1)
    vis.publish(bboxes=bounding_boxes, pcloud=pcl_data[mask], object_vectors=output[mask])
