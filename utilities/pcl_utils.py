import numpy as np


def rotate_pcl(pcl, angle):
    sin = np.sin(angle)
    cos = np.cos(angle)
    rotation_matrix_trp = np.array(((cos, sin, 0),
                                    (-sin, cos, 0),
                                    (0, 0, 1)))
    rotated_pcl = np.matmul(pcl, rotation_matrix_trp)
    return rotated_pcl
