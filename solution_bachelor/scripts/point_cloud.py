import rclpy
from rclpy.node import Node
import sensor_msgs.msg
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import open3d as o3d

class PointCloudSaver(Node):
    def __init__(self):
        super().__init__('pointcloud_saver')
        self.subscription = self.create_subscription(
            sensor_msgs.msg.PointCloud2,
            '/camera/depth/points',
            self.callback,
            10)
        self.subscription  # avoid unused variable warning

    def callback(self, msg):
        self.get_logger().info("Received PointCloud2 message")
        points = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)))

        # Tạo Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Lưu thành file PCD
        o3d.io.write_point_cloud("dock/output.pcd", pcd)
        self.get_logger().info("Saved point cloud to output.pcd")
        rclpy.shutdown()  # Dừng node sau khi ghi

def main(args=None):
    rclpy.init(args=args)
    node = PointCloudSaver()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
