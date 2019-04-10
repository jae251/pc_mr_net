import numpy as np
from utilities.pcl_utils import rotate_pcl


class BoundingBox:
    def __init__(self, id, label, x, y, z, l, b, h, angle):
        self.id = id
        self.label = label
        self.pos = np.array((x, y, z))
        self.size = np.array((l, b, h))
        self.angle = angle
        self.l = 0.5 * l * np.array((np.cos(angle), np.sin(angle), 0))
        self.b = 0.5 * b * np.array((-np.sin(angle), np.cos(angle), 0))
        self.h = np.array((0, 0, 0.5 * h))
        self.support_vector_matrix = np.dstack((self.l, self.b, self.h))[0]

        self.color = (1, 1, 1)
        self.frame_id = "map"

    @classmethod
    def from_kitti_string(cls, id, kitti_string, tf_matrix=None):
        # parse one line in KITTI label file
        # tf_matrix should be given, since kitti objects are given in camera coordinate system, not in lidar cs
        items = kitti_string.split()
        items = [items[0]] + [float(v) for v in items[1:]]
        label, truncated, occluded, alpha, x1, y1, x2, y2, xd, yd, zd, x, y, z, roty = items
        if tf_matrix is not None:
            p = np.array((x, y, z, 1), dtype=np.float32)
            x, y, z, _ = np.matmul(p, tf_matrix)
        z += xd * .5
        roty *= -1
        return cls(id, label, x, y, z, yd, zd, xd, roty)

    @classmethod
    def from_numpy(cls, numpy_entry):
        id = numpy_entry["id"]
        label = numpy_entry["label"]
        x, y, z = numpy_entry["position"]
        l, b, h = numpy_entry["size"]
        angle = numpy_entry["angle"]
        return cls(id, label, x, y, z, l, b, h, angle)

    def to_numpy(self):
        return np.array((self.id,
                         self.label,
                         self.pos,
                         self.size,
                         self.angle), dtype=np.dtype([("id", np.uint16),
                                                      ("label", (np.string_, 16)),
                                                      ("position", (np.float64, 3)),
                                                      ("size", (np.float64, 3)),
                                                      ("angle", np.float64)]))

    def check_points_inside(self, pcl):
        linear_combination = np.linalg.solve(self.support_vector_matrix, (pcl - self.pos).transpose())
        inside_box = np.all(linear_combination <= 1, axis=0) * np.all(linear_combination >= -1, axis=0)
        return inside_box

    def tf_into_bbox_cs(self, pcl):
        # transform a point cloud into bounding box coordinate system
        # with bounding box position as origin and the same orientation
        return rotate_pcl(pcl - self.pos, self.angle)

    def compute_regression_labels(self, pcl):
        # return relative coordinates of points for points inside bounding box, else 0
        regression_labels = self.tf_into_bbox_cs(pcl)
        mask = (regression_labels > self.size) + (regression_labels < -self.size)
        regression_labels[np.any(mask, axis=1)] = 0
        return regression_labels

    def get_polygon_2d(self):
        from shapely.geometry import Polygon
        polygon = Polygon(self.get_corner_points_2d())
        return polygon

    def get_corner_points(self):
        corner_points = self.pos + np.vstack(((self.l + self.b - self.h),
                                              (self.l - self.b - self.h),
                                              (-self.l + self.b - self.h),
                                              (-self.l - self.b - self.h),
                                              (self.l + self.b + self.h),
                                              (self.l - self.b + self.h),
                                              (-self.l + self.b + self.h),
                                              (-self.l - self.b + self.h)))
        return corner_points

    def get_corner_points_2d(self):
        corner_points = self.pos + np.vstack(((self.l + self.b),
                                              (self.l - self.b),
                                              (-self.l - self.b),
                                              (-self.l + self.b)))

        return corner_points

    def copy(self):
        return BoundingBox(self.id, self.label, *self.pos, *self.size, self.angle)

    ####################################################################################################################
    # methods for visualization #
    #############################
    def draw_to_plot_2d(self, plot):
        corner_points_2d = self.get_corner_points_2d()
        polygon = np.vstack((corner_points_2d, corner_points_2d[0]))
        plot.plot(polygon[:, 0], polygon[:, 1])
        return plot

    def to_rviz_marker(self):
        from pyquaternion import Quaternion
        from visualization_msgs.msg import Marker
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.type = marker.CUBE
        marker.action = marker.ADD
        marker.id = self.id
        marker.scale.x = self.size[0]
        marker.scale.y = self.size[1]
        marker.scale.z = self.size[2]
        marker.pose.position.x = self.pos[0]
        marker.pose.position.y = self.pos[1]
        marker.pose.position.z = self.pos[2]
        orientation = Quaternion(axis=(0, 0, 1), angle=self.angle)
        marker.pose.orientation.x = orientation.x
        marker.pose.orientation.y = orientation.y
        marker.pose.orientation.z = orientation.z
        marker.pose.orientation.w = orientation.w
        marker.color.a = .3

        marker.color.r = self.color[0]
        marker.color.g = self.color[1]
        marker.color.b = self.color[2]

        text = Marker()
        text.header.frame_id = self.frame_id
        text.type = text.TEXT_VIEW_FACING
        text.action = text.ADD
        text.id = self.id
        text.color.a = 1
        text.color.r = 0
        text.color.g = 0
        text.color.b = 0
        text.scale.z = .3
        text.pose.position.x = self.pos[0]
        text.pose.position.y = self.pos[1]
        text.pose.position.z = self.pos[2]
        text.text = "ID: {}\nLabel: {}".format(self.id, self.label)
        text.ns = "basic_shapes"

        return marker, text
