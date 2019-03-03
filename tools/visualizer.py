'''
Example usage:

from visualization.visualizer import Visualizer
viz = Visualizer()
viz.publish(bboxes=bboxes, pcloud=pcloud, frame_id="root")

'''
import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
import std_msgs.msg


class Visualizer:
    def __init__(self, pcloud_topic="pcloud", bbox_topic="bboxes", frame_id="root"):
        rospy.init_node("pcloud_publisher")
        self.bb_pub = rospy.Publisher(bbox_topic, MarkerArray, queue_size=10)
        self.pc_pub = rospy.Publisher(pcloud_topic, PointCloud2, queue_size=10)
        self.frame_id = frame_id

        while True:
            if self.bb_pub.get_num_connections() > 0:
                if self.pc_pub.get_num_connections() > 0:
                    break

    def publish(self, bboxes=None, pcloud=None, pcloud_color=None):
        if pcloud is not None:
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

            for bbox in bboxes:
                marker, text = bbox.to_rviz_marker()
                bb.markers.append(text)
                bb.markers.append(marker)
            self.bb_pub.publish(bb)
