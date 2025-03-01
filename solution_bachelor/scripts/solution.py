#!/usr/bin/env python3

import numpy as np
import os

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, PointCloud2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

import cv2
from cv_bridge import CvBridge, CvBridgeError

from vision import detect_lane
import open3d as o3d
import sensor_msgs_py.point_cloud2 as pc2

class SimpleMover(Node):

    def __init__(self, rate=10):
        super().__init__('solution')
        self.get_logger().info("...solution node started2 ..................")

        self.gui = os.getenv('GUI') == 'true' or os.getenv('GUI') == 'True'

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.vision_pub = self.create_publisher(Image, '/vision', 10)
        self.ransac_pub = self.create_publisher(PointCloud2, '/ransac_points', 10)

        self.create_subscription(Image, '/camera/image_raw', self.camera_cb, 10)
        self.create_subscription(Odometry, '/odom', self.gps_cb, 10)
        self.create_subscription(PointCloud2, '/camera/points', self.points_cb, 10)

        self.cv_bridge = CvBridge()

        # timer calls move function at the specified rate
        self.rate = rate  # Hz
        timer_period = 1.0 / float(self.rate)
        self.timer = self.create_timer(timer_period, self.move)

        # Control constants
        self.target_speed = 1.5
        self.speed_gain = 20
        self.steering_gain = 0.008

        # Received data
        self.current_speed = 0.0
        self.object_center_x = None
        self.end_points = None
        self.last_valid_end_points = None

        # Moving average filter parameters
        self.linear_cmd_history = []
        self.steering_angle_history = []
        self.filter_size = 100  # Number of values to average over

        # states
        self.simulation_started = False

    def camera_cb(self, msg):
        # try:
        #     cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        #     line_image, self.end_points = detect_lane(cv_image)
        #     self.get_logger().info(f"end points: {self.end_points}")

        #     if self.end_points is not None:
        #         if len(self.end_points) == 2:
        #             left, right = self.end_points
        #             if left is None and right is None:
        #                 self.object_center_x = None
        #             self.object_center_x = (left[0] + right[0]) / 2
        #             self.last_valid_end_points = self.end_points
        #         self.get_logger().info(f"End points: {self.end_points}")
        #     else:
        #         if self.last_valid_end_points is not None:
        #             left, right = self.last_valid_end_points
        #             self.object_center_x = (left[0] + right[0]) / 2
        #         else:
        #             self.object_center_x = None
        #         self.get_logger().warn("No valid end points detected.")

        #     # Show the final result
        #     if self.gui:
        #         vision_msg = self.cv_bridge.cv2_to_imgmsg(line_image, "bgr8")
        #         self.vision_pub.publish(vision_msg)
        #         if cv2.waitKey(1) & 0xFF == ord('q'):
        #             cv2.destroyAllWindows()
        #             self.get_logger().info("Video writer released and window closed.")

        # except CvBridgeError as e:
        #     self.get_logger().error(f"cv_bridge error: {e}")
        return

    def gps_cb(self, msg):
        # self.get_logger().info(f"The position of robot is: {msg.pose.pose.position.x}, {msg.pose.pose.position.y}")
        self.current_speed = msg.twist.twist.linear.x
        # self.get_logger().info(f"Current speed: {self.current_speed}")

    def points_cb(self, msg):
        def random_downsample(pcd, target_size=50000):
            """
            Giảm số lượng điểm bằng cách chọn ngẫu nhiên target_size điểm.
            """
            total_points = len(pcd.points)
            if total_points <= target_size:
                return pcd  # Không cần giảm nữa

            indices = np.random.choice(total_points, target_size, replace=False)  # Chọn ngẫu nhiên
            down_pcd = pcd.select_by_index(indices)
            
            print(f"Trước: {total_points} điểm, Sau: {len(down_pcd.points)} điểm")
            return down_pcd
        
        # Convert ROS PointCloud2 message to numpy array
        points_list = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        if len(points_list) == 0:
            return

        # Extract x, y, z coordinates and convert to numpy array
        points = np.array([[p[0], p[1], p[2]] for p in points_list], dtype=np.float32)

        print('Shape origin:', points.shape)
        # Reduce the size of the point cloud by downsampling
        voxel_size = 10  # Adjust voxel size as needed
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        downsampled_pcd = random_downsample(pcd)
        print('Reduced size:', len(downsampled_pcd.points))

        # Apply RANSAC to filter the point cloud
        plane_model, inliers = downsampled_pcd.segment_plane(distance_threshold=0.5,
                                                             ransac_n=3,
                                                             num_iterations=200)

        inlier_cloud = downsampled_pcd.select_by_index(inliers)

        """ Chiếu Point Cloud xuống mặt phẳng 2D """
        points = np.asarray(inlier_cloud.points)

        # Ảnh 500x500 để chứa dữ liệu 2D
        img = np.zeros((500, 500), dtype=np.uint8)
        
        for point in points:
            x, y, z = point
            # print(x, y, z)

            # Loại bỏ điểm quá cao/thấp (giới hạn mặt đường)
            if z > -1 and z < 20:  
                img_x = int(x * 50 + 250)  # Scale to image
                img_y = int(y * 50 + 250)

                if 0 <= img_x < 500 and 0 <= img_y < 500:
                    img[img_y, img_x] = 255  # Đánh dấu điểm trắng

        cv2.imshow('RANSAC', img)
        cv2.waitKey(0)

        # Convert Open3D PointCloud to ROS PointCloud2 message
        ransac_points_msg = pc2.create_cloud_xyz32(msg.header, np.asarray(inlier_cloud.points))

        # Publish the filtered point cloud
        self.ransac_pub.publish(ransac_points_msg)

    def move(self):
        # Control algorithm
        # Nếu chưa nhận được dữ liệu đối tượng thì không xử lý
        if self.object_center_x is None:
            return

        # Tâm khung hình với chiều rộng 640px là 320px
        desired_center = 320.0
        error_x = desired_center - self.object_center_x

        # Tính góc lái: sử dụng controller tỷ lệ
        steering_angle = error_x * self.steering_gain
        # Giới hạn góc lái trong khoảng [-0.6, 0.6] rad
        steering_angle = max(min(steering_angle, 0.6), -0.6)

        # Controller tốc độ:
        # target_speed được đặt là 1 m/s
        speed_error = self.target_speed - self.current_speed
        # Sử dụng controller tỷ lệ cho tốc độ:
        # Khi lên dốc, current_speed thấp nên speed_error dương => linear_cmd dương (tăng lực kéo)
        # Khi xuống dốc, current_speed cao nên speed_error âm => linear_cmd âm (hãm phanh)
        linear_cmd = self.speed_gain * speed_error
        # Giới hạn linear_cmd trong khoảng [-1, 1]
        linear_cmd = max(min(linear_cmd, 15), -15)

        # Apply moving average filter
        self.linear_cmd_history.append(linear_cmd)
        self.steering_angle_history.append(steering_angle)

        if len(self.linear_cmd_history) > self.filter_size:
            self.linear_cmd_history.pop(0)
        if len(self.steering_angle_history) > self.filter_size:
            self.steering_angle_history.pop(0)

        filtered_linear_cmd = sum(self.linear_cmd_history) / len(self.linear_cmd_history)
        filtered_steering_angle = sum(self.steering_angle_history) / len(self.steering_angle_history)

        # Tạo thông điệp Twist: linear.x là % mômen, angular.z là góc lái (radian)
        twist_msg = Twist()
        twist_msg.linear.x = float(filtered_linear_cmd)  # Ensure linear_cmd is a float
        twist_msg.angular.z = float(filtered_steering_angle)  # Ensure steering_angle is a float

        self.cmd_vel_pub.publish(twist_msg)
        self.get_logger().info(
            f"Xuất lệnh: tốc độ (%% mômen) = {twist_msg.linear.x:.2f}, góc lái = {twist_msg.angular.z:.2f} rad"
        )


def main(args=None):
    rclpy.init(args=args)
    node = SimpleMover(rate=30)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()