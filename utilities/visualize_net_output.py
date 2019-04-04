import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import std_msgs.msg


class Visualizer:
    def __init__(self, pcloud_topic="point_cloud", bbox_topic="bounding_boxes", object_vector_topic="object_vectors",
                 frame_id="map", box_color=(1, 1, 1)):
        rospy.init_node("net_visualizer")
        self.bb_pub = rospy.Publisher(bbox_topic, MarkerArray, queue_size=10)
        self.pc_pub = rospy.Publisher(pcloud_topic, PointCloud2, queue_size=10)
        self.ov_pub = rospy.Publisher(object_vector_topic, MarkerArray, queue_size=10)
        self.frame_id = frame_id
        self.box_color = box_color

        while True:
            if self.bb_pub.get_num_connections() > 0:
                if self.pc_pub.get_num_connections() > 0:
                    if self.ov_pub.get_num_connections() > 0:
                        break

    def publish(self, bboxes=None, pcloud=None, pcloud_color=None, object_vectors=None, object_vector_colors=None):
        if pcloud is not None:
            pcloud = pcloud.astype(np.float32)
            pc = PointCloud2()
            pc.header = std_msgs.msg.Header()
            pc.header.frame_id = self.frame_id
            pc.header.stamp = rospy.Time.now()

            l = len(pcloud)
            pc.height = 1
            pc.width = l
            pc.fields = [PointField('x', 0, PointField.FLOAT32, 1),
                         PointField('y', 4, PointField.FLOAT32, 1),
                         PointField('z', 8, PointField.FLOAT32, 1),
                         PointField('intensities', 12, PointField.FLOAT32, 1)]
            pc.is_bigendian = False
            pc.point_step = 16
            pc.row_step = 16 * l
            if pcloud_color is None:
                pcl_data = np.hstack((pcloud[:, :3], np.ones((len(pcloud), 1), dtype=np.float32)))
            else:
                pcl_data = np.hstack((pcloud[:, :3], pcloud_color.reshape(-1, 1)))
            pc.data = pcl_data.tostring()
            self.pc_pub.publish(pc)

        if bboxes is not None:
            bb = MarkerArray()
            marker = Marker()
            marker.action = marker.DELETEALL
            bb.markers.append(marker)

            for box in bboxes:
                marker, text = box.to_rviz_marker()
                bb.markers.append(text)
                bb.markers.append(marker)
            self.bb_pub.publish(bb)

        if object_vectors is not None:
            assert pcloud is not None
            ov = MarkerArray()
            marker = Marker()
            marker.action = marker.DELETEALL
            ov.markers.append(marker)
            if object_vector_colors is None:
                object_vector_colors = DummyArray((0, 1, 0))
            elif isinstance(object_vector_colors, tuple):
                object_vector_colors = DummyArray(object_vector_colors)
            for c, vec in enumerate(object_vectors):
                marker = Marker()
                marker.header.frame_id = self.frame_id
                marker.type = marker.ARROW
                marker.action = marker.ADD
                marker.id = c
                marker.scale.x = .01
                marker.scale.y = .025
                marker.scale.z = .1
                color = object_vector_colors[c]
                marker.color.a = 1
                marker.color.r = color[0]
                marker.color.g = color[1]
                marker.color.b = color[2]
                marker.points = [Point(*tuple(pcloud[c])), Point(*tuple(vec))]
                ov.markers.append(marker)
            self.ov_pub.publish(ov)


class DummyArray:
    # the purpose of this class is to recieve an index argument and discard it,
    # always returning the value it was initialized with
    def __init__(self, item):
        self.item = item

    def __getitem__(self, item):
        return self.item


if __name__ == '__main__':
    vis = Visualizer()
    vis.publish(pcloud=np.random.uniform(-1, 1, (10, 3)).astype(np.float32),
                object_vectors=np.random.uniform(-1, 1, (10, 3)))
