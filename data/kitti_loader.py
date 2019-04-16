import os
import numpy as np

try:
    from utilities.bounding_boxes import BoundingBox
except ImportError:
    from ..utilities.bounding_boxes import BoundingBox
try:
    from hdf_dataset_loader import HdfDataset
except ImportError:
    from .hdf_dataset_loader import HdfDataset


class KittiLoader(HdfDataset):
    def __init__(self, dataset_folder=os.path.expanduser("~/Downloads/KITTI/training")):
        self.kitti_path = dataset_folder
        self.pcl_data_dir = os.path.join(dataset_folder, "velodyne")
        self.label_data_dir = os.path.join(dataset_folder, "label_2")
        self.calib_data_dir = os.path.join(dataset_folder, "calib")
        self.pcl_frames = os.listdir(self.pcl_data_dir)
        self.label_frames = os.listdir(self.label_data_dir)
        self.calib_frames = os.listdir(self.calib_data_dir)

    @staticmethod
    def _read_pcl_file(file):
        data = np.fromfile(file, dtype=np.float32).reshape(-1, 4)
        return data

    @staticmethod
    def _read_label_file(file, tf_matrix=None):
        objects = []
        with open(file) as f:
            for i, line in enumerate(f):
                obj = BoundingBox.from_kitti_string(i, line, tf_matrix)
                objects.append(obj)
        return objects

    @staticmethod
    def _read_calib_file(file):
        with open(file) as f:
            for line in f:
                try:
                    key, values = line.split(":", 1)
                except ValueError:
                    break
                if key == "R0_rect":
                    r0 = np.array([float(s) for s in values.split()]).reshape(3, 3)
                    r0 = np.vstack((np.hstack((r0, np.array((0, 0, 0)).reshape(3, 1))), (0, 0, 0, 1)))
                elif key == "Tr_velo_to_cam":
                    trvtc = np.array([float(s) for s in values.split()]).reshape(3, 4)
                    trvtc = np.vstack((trvtc, (0, 0, 0, 1)))
        tf_matrix = np.linalg.inv(np.matmul(r0, trvtc).transpose())
        return tf_matrix

    def __len__(self):
        return len(self.pcl_frames)

    def __getitem__(self, item):
        pcl_data = self._read_pcl_file(self.pcl_data_dir + "/" + self.pcl_frames[item])
        pcl_data = self.reconstruct_picture_format(pcl_data)
        tf_matrix = self._read_calib_file(self.calib_data_dir + "/" + self.calib_frames[item])
        bounding_box_data = self._read_label_file(self.label_data_dir + "/" + self.label_frames[item], tf_matrix)
        feature_vector = self.compute_feature_vector(pcl_data)
        label_vector = self.compute_label_vector(pcl_data, bounding_box_data)
        if self.transform:
            feature_vector, label_vector = self.transform(feature_vector, label_vector)
        return feature_vector, label_vector

    def get_raw_data(self, item):
        pcl_data = self._read_pcl_file(self.pcl_data_dir + "/" + self.pcl_frames[item])
        tf_matrix = self._read_calib_file(self.calib_data_dir + "/" + self.calib_frames[item])
        bounding_box_data = self._read_label_file(self.label_data_dir + "/" + self.label_frames[item], tf_matrix)
        return pcl_data, bounding_box_data

    def reconstruct_picture_format(self, pcl):
        raise NotImplementedError


if __name__ == '__main__':
    kitti_path = "/mnt/c/Users/usr/Downloads/KITTI/training"
    # kitti_path = os.path.expanduser("~/Downloads/KITTI/training")
    data_loader = KittiLoader(kitti_path)
    from utilities.rviz_visualization import Visualizer

    vis = Visualizer()
    pcl, bboxes = data_loader.get_raw_data(8)
    vis.publish(pcloud=pcl, bboxes=bboxes)
