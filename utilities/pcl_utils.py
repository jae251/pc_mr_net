import numpy as np
from scipy.spatial.distance import pdist, squareform


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
    distance_matrix = squareform(pdist(pcl))
    sorted_indices_by_distance = np.argsort(distance_matrix, axis=1)
    nearest_neighbour_indices = sorted_indices_by_distance[:, 1:n + 1]  # first entry is self distance
    row_indices = np.repeat(sorted_indices_by_distance[:, 0][:, np.newaxis], n, axis=1)

    nearest_points = pcl[nearest_neighbour_indices]
    # print(nearest_points.shape)
    distances = distance_matrix[row_indices, nearest_neighbour_indices]
    # print(distances.shape)
    return nearest_points, distances


def find_points_inside_distance(pcl, d):
    '''
    for each point in pcl find all other points inside distance d
    '''
    distance_matrix = squareform(pdist(pcl))
    cutoff_index = np.sum(distance_matrix <= d, axis=1)
    sorted_indices = np.argsort(distance_matrix, axis=1)
    points_inside_distance = []
    for i, indices in enumerate(sorted_indices):
        points = pcl[sorted_indices[i, 1:cutoff_index[i]]]
        points_inside_distance.append(points)
    return points_inside_distance
