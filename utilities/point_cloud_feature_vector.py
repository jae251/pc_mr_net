import numpy as np

from utilities.pcl_utils import find_nearest_points


class CreateFeatureVector:
    '''
    After setting parameters this class creates a set of feature vectors from a point cloud, over which
    convolution should be applied. A feature vector contains for every point P in the point cloud the
    relative coordinates between P and a fixed number of points, if their distance is lower than a threshold.
     The distance between these points is appended as well.
    '''

    def __init__(self, max_nr_nearest_neighbours, max_distance):
        self.max_n = max_nr_nearest_neighbours
        self.max_dist = max_distance

    def __call__(self, pcl):
        nearest_points, distances = find_nearest_points(pcl, self.max_n)
        mask = distances <= self.max_dist
        nearest_points[mask] = 0
        distances[mask] = 0
        feature_vector = np.append(nearest_points.reshape(-1, self.max_n), distances)
        return feature_vector
