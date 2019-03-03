import numpy as np
from scipy.spatial import distance


def rotate_pcl(pcl, angle):
    sin = np.sin(angle)
    cos = np.cos(angle)
    rotation_matrix_trp = np.array(((cos, sin, 0),
                                    (-sin, cos, 0),
                                    (0, 0, 1)))
    rotated_pcl = np.matmul(pcl, rotation_matrix_trp)
    return rotated_pcl


def find_nearest_points(pcl, n):
    '''
    find indices and distances of n nearest points in point cloud
    :param pcl: point cloud array
    :param n: number of nearest neighbours to find
    :return: array of nearest points, shape=(len(pcl),n,3),
             array of distances, shape=(len(pcl),n)
    '''
    distance_matrix = distance.squareform(distance.pdist(pcl))
    sorted_indices_by_distance = np.argsort(distance_matrix, axis=1)
    nearest_neighbour_indices = sorted_indices_by_distance[:, 1:n + 1]  # first entry is self distance
    row_indices = np.repeat(sorted_indices_by_distance[:, 0][:, np.newaxis], n, axis=1)

    nearest_points = pcl[nearest_neighbour_indices]
    # print(nearest_points.shape)
    distances = distance_matrix[row_indices, nearest_neighbour_indices]
    # print(distances.shape)
    return nearest_points, distances
