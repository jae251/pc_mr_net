import numpy as np
import os
from PIL import Image
from h5py import File

from lidar_simulation.lidar import Lidar
from lidar_simulation.data_loaders.load_3d_models import load_obj_file
from lidar_simulation.utilities.geometry_calculations import rotate_point_cloud


class LidarDistancePicture(Lidar):
    def create_rays(self, vertices):
        vertices_spherical = self._tf_into_spherical_sensor_coordinates(vertices)
        azimuth_min, azimuth_max, elevation_min, elevation_max = self._model_view_dimensions(vertices_spherical)
        n1 = int((azimuth_max - azimuth_min) / self.delta_azimuth)
        n2 = int((elevation_max - elevation_min) / self.delta_elevation)
        az = np.linspace(azimuth_min, azimuth_max, n1)
        el = np.linspace(elevation_min, elevation_max, n2)
        rays_spherical = np.dstack((np.ones(n1 * n2), *[m.ravel() for m in np.meshgrid(az, el)]))[0]
        ray_directions = self._tf_into_cartesian_coordinates(rays_spherical)
        self.picture_dimensions = (len(el), len(az))
        return self.position, ray_directions

    def reshape_to_picture(self, arr):
        try:
            return arr.reshape(*self.picture_dimensions)
        except ValueError:
            return arr.reshape(*self.picture_dimensions, -1)


obj_file = os.path.expanduser("~/Downloads/3d models/Porsche_911_GT2.obj")
vertices, polygons, uv_coordinates, uv_coordinate_indices = load_obj_file(obj_file, texture=True)
vertices = rotate_point_cloud(vertices, -.5)
lidar = LidarDistancePicture(delta_azimuth=2 * np.pi / 3000,
                             delta_elevation=np.pi / 800,
                             position=(0, -10, 1))
point_cloud, ray_hit_uv = lidar.sample_3d_model_with_texture_gpu(vertices,
                                                                 polygons,
                                                                 uv_coordinates,
                                                                 uv_coordinate_indices)
# TODO: Lidar should also return the polygon index of the ray hit for reconstruction of object index
print(lidar.picture_dimensions)
point_cloud[np.any(point_cloud != 0, axis=1)] -= np.array((0, -10, 1))
distances = np.linalg.norm(point_cloud, axis=1)

distance_picture = lidar.reshape_to_picture(distances)
uv_coordinates = lidar.reshape_to_picture(ray_hit_uv)

# import pptk
# v = pptk.viewer(point_cloud[np.any(point_cloud != 0, axis=1)])
# v.set(point_size=.003)

mock_object_mask = (distance_picture != 0).astype(np.uint16)

with File("sample_data.hdf5", "w") as f:
    f.create_dataset("distance_picture", data=distance_picture)
    f.create_dataset("uv_coordinates", data=uv_coordinates)
    f.create_dataset("object_mask", data=mock_object_mask)

maximum = distances.max()
minimum = distances[distances != 0].min()
print(maximum)
print(minimum)
distances[distances != 0] -= minimum
distances *= 255 / (maximum - minimum)
im = Image.fromarray(lidar.reshape_to_picture(distances)[::-1, ::-1])
im.show()
