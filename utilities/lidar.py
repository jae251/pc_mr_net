import numpy as np

from lidar_simulation.lidar import Lidar


class Lidar(Lidar):
    '''
    This modifies the Lidar class so that a distance picture can be calculated by reshaping the sampled points
    with the self.sampling_dimensions attribute
    '''
    def create_rays(self, vertices):
        vertices_spherical = self._tf_into_spherical_sensor_coordinates(vertices)
        azimuth_min, azimuth_max, elevation_min, elevation_max = self._model_view_dimensions(vertices_spherical)
        n1 = int((azimuth_max - azimuth_min) / self.delta_azimuth)
        n2 = int((elevation_max - elevation_min) / self.delta_elevation)
        az = np.linspace(azimuth_min, azimuth_max, n1)
        el = np.linspace(elevation_min, elevation_max, n2)
        rays_spherical = np.dstack((np.ones(n1 * n2), *[m.ravel() for m in np.meshgrid(az, el)]))[0]
        ray_directions = self._tf_into_cartesian_coordinates(rays_spherical)
        self.sampling_dimensions = (len(el), len(az))
        return self.position, ray_directions
