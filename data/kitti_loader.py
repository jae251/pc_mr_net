import os
import numpy as np

from bounding_boxes import BoundingBox


class KittiLoader:
    def __init__(self, kitti_path=os.path.expanduser("~/Downloads/KITTI")):
        self.kitti_path = kitti_path
        self.training_data_path = kitti_path + "/training/velodyne"
        # self.testing_data_path = kitti_path + "/testing/velodyne"
        self.training_label_path = kitti_path + "/training/label_2"
        self.calib_path = kitti_path + "/training/calib"
        self.pcl_frames = os.listdir(self.training_data_path)
        self.label_frames = os.listdir(self.training_label_path)
        self.calib_frames = os.listdir(self.calib_path)

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

    def __iter__(self):
        self._order = np.random.shuffle(range(len(self.pcl_frames)))
        self.iteration = 0
        return self

    # def __next__(self):
    def next(self):
        try:
            pcl_data = self._read_pcl_file(self.training_data_path + "/" + self.pcl_frames[self.iteration])
            tf_matrix = self._read_calib_file(self.calib_path + "/" + self.calib_frames[self.iteration])
            label_data = self._read_label_file(self.training_label_path + "/" + self.label_frames[self.iteration],
                                               tf_matrix)
        except IndexError:
            raise StopIteration
        self.iteration += 1
        return pcl_data, label_data


def add_marker(pcl, rgb, x, y):
    debug_pole = np.hstack((np.repeat(np.array((x, y)).reshape(1, 2), 100, axis=0),
                            np.linspace(-5, 5, 100).reshape(-1, 1)))
    rgb = np.vstack((rgb, np.repeat(np.array((1, 0, 0)).reshape(1, 3), len(debug_pole), axis=0)))
    pcl = np.vstack((pcl, debug_pole))
    return pcl, rgb


if __name__ == '__main__':
    data_loader = KittiLoader("/cygdrive/c/Users/usr/Downloads/KITTI")
    from tools.visualizer import Visualizer
    from time import sleep

    vis = Visualizer()
    for pcl, bboxes in data_loader:
        vis.publish(pcloud=pcl, bboxes=bboxes)
        print("Published!")
        sleep(3)
