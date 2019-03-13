import numpy as np
from PIL import Image
from h5py import File
from pptk import viewer


def visualize(filepath):
    with File(filepath, "r") as f:
        pcl = np.array(f["point_cloud"])
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


if __name__ == '__main__':
    filename = "../data/dataset_one_car/train/00000.hdf5"
    visualize(filename)
