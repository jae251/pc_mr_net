import numpy as np
import os
from h5py import File

from utilities.lidar import Lidar
from lidar_simulation.data_loaders.load_3d_models import load_obj_file
from lidar_simulation.scene_builder import Scene

from utilities.utils import adapt_z_position

if not os.path.isdir("dataset_one_car"):
    os.mkdir("dataset_one_car")
if not os.path.isdir("dataset_one_car/train"):
    os.mkdir("dataset_one_car/train")
if not os.path.isdir("dataset_one_car/eval"):
    os.mkdir("dataset_one_car/eval")

scene = Scene(spawn_area=((-20, -5), (20, -5), (20, 20), (-20, 20)))
obj_file = os.path.expanduser("~/Downloads/3d models/Porsche_911_GT2.obj")
# obj_file = "/mnt/sda2/Users/usr/Downloads/3d models/Porsche_911_GT2.obj"
scene.add_model_to_shelf(*load_obj_file(obj_file), "Car")

nr_of_scenes = 12000
training_samples = 10000
for scene_nr in range(nr_of_scenes):
    print(scene_nr)
    if scene_nr < training_samples:
        folder = "train"
    else:
        folder = "eval"
    filename = "dataset_one_car/{}/{:>05}.hdf5".format(folder, scene_nr)
    if os.path.isfile(filename):
        continue

    scene.clear()
    scene.place_object_randomly("Car")
    vertices, polygons = scene.build_scene()
    h = np.random.uniform(0, 3)
    lidar = Lidar(delta_azimuth=2 * np.pi / 3000,
                  delta_elevation=np.pi / 800,
                  position=(0, -10, h))
    pcl = lidar.sample_3d_model_gpu(vertices, polygons)
    width, height = lidar.sampling_dimensions
    pcl = pcl.reshape(height, width, 3)
    bounding_boxes = adapt_z_position(scene.get_bounding_boxes())

    with File(filename, "w") as f:
        f.create_dataset("point_cloud", data=pcl)
        f.create_dataset("bounding_boxes", data=bounding_boxes)
