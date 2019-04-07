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
            bounding_box_data = np.array(f["bounding_boxes"])
        bounding_boxes = []
        for bb in bounding_box_data:
            bounding_boxes.append(BoundingBox.from_numpy(bb))
        feature_vector = self.compute_feature_vector(pcl_data)
        label_vector = self.compute_label_vector(pcl_data, bounding_box_data)
        return feature_vector, pcl_data.reshape(-1, 3), bounding_boxes, label_vector


MODEL_DIR = "net_weights"
DATA_DIR = "../../data/dataset_one_car/train"
# DATA_DIR = "../../data/dataset_one_car/eval"


if __name__ == '__main__':
    net = PointCloudMapRegressionNet()
    model = os.path.join(MODEL_DIR, sorted(os.listdir(MODEL_DIR))[-1])
    net.load_state_dict(torch.load(model, map_location="cpu"))

    data_loader = InferenceDataset(DATA_DIR)
    feature_vector, pcl_data, bounding_boxes, label_vector = data_loader[0]  # np.random.randint(0, len(data_loader))]
    labels = np.rollaxis(label_vector.numpy()[0], 0, 3).reshape(-1, 3)
    output = np.rollaxis(net(feature_vector).data.numpy()[0], 0, 3).reshape(-1, 3)
    print(np.linalg.norm(output - labels))
    print(np.sum((output - labels) ** 2) * .5)
    vis = Visualizer()
    # mask = np.any(pcl_data != 0, axis=1) #* np.any(labels != 0, axis=1)  # * np.any(output != 0, axis=1)
    # pcl = pcl_data[mask]
    # vis.publish(bboxes=bounding_boxes, pcloud=pcl, object_vectors=np.zeros_like(pcl) + bounding_boxes[0].pos)
    vis.publish(bboxes=bounding_boxes, pcloud=pcl_data, object_vectors=pcl_data - output)
    # vis.publish(bboxes=bounding_boxes, pcloud=pcl_data)
