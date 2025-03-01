import open3d as o3d
import numpy as np
import os

# Load point cloud from file
pcd = o3d.io.read_point_cloud("/home/toan/Documents/bachelor/image_processing/output.pcd")

# Convert to NumPy
points = np.asarray(pcd.points)

# information
print(pcd) # PointCloud with 307200 points.
print(points.shape) # (307200, 3)

# Kiểm tra NaN hoặc Inf
print("Contains NaN:", np.isnan(points).any())
print("Contains Inf:", np.isinf(points).any())

# Loại bỏ NaN và Inf
filtered_points = points[~np.isnan(points).any(axis=1)]
filtered_points = filtered_points[~np.isinf(filtered_points).any(axis=1)]

# Gán lại vào point cloud
pcd.points = o3d.utility.Vector3dVector(filtered_points)


# Filter by z-axis
z_threshold = 7
filtered_points = filtered_points[filtered_points[:, 2] > z_threshold]
pcd.points = o3d.utility.Vector3dVector(filtered_points)

# plane_model, inliers = pcd.segment_plane(distance_threshold=1, ransac_n=3, num_iterations=1000)
# inlier_cloud = pcd.select_by_index(inliers)



# # Tạo trục tọa độ gốc (dài 1 đơn vị)
axis_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10., origin=[0, 0, 0])

# Hiển thị point cloud cùng trục tọa độ
o3d.visualization.draw_geometries([pcd, axis_frame])
