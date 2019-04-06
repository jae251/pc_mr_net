import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from h5py import File
from pptk import viewer
from utilities.bounding_boxes import BoundingBox


def visualize(filepath):
    with File(filepath, "r") as f:
        pcl = np.array(f["point_cloud"])
        bb = BoundingBox.from_numpy(f["bounding_boxes"][0])
    print(pcl.shape)
    viewer(pcl)
    distance_picture = np.linalg.norm(pcl, axis=2)
    distance_picture = distance_picture[::-1]  # switch y-axis, because PIL uses different coordinate system
    min_distance = distance_picture[distance_picture != 0].min()
    distance_picture[distance_picture != 0] -= min_distance
    max_distance = distance_picture.max()
    normalize_coeff = 255 / max_distance
    distance_picture *= normalize_coeff
    im = Image.fromarray(distance_picture)
    im.show()
    bb.draw_to_plot_2d(plt)
    pcl = pcl.reshape(-1, 3)
    plt.scatter(pcl[:, 0], pcl[:, 1], c="g", s=.1)
    plt.gca().set_aspect(aspect=1)
    plt.show()


if __name__ == '__main__':
    filename = "../data/dataset_one_car/train/00000.hdf5"
    visualize(filename)
