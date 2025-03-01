# Định nghĩa tọa độ và tỷ lệ co lại
points = [(258, 188), (212, 373)]
scale_x = 640 / 469
scale_y = 480 / 623

# Tính tọa độ ngược cho mỗi điểm
reversed_points = [(int(x * scale_x), int(y * scale_y)) for x, y in points]

# Tính điểm đối xứng theo trục x
symmetric_points = [(640 - x, y) for x, y in points]

# Tính tọa độ ngược cho các điểm đối xứng
reversed_symmetric_points = [(int(x * scale_x), int(y * scale_y)) for x, y in symmetric_points]

# In kết quả
for original, reversed_point, symmetric_point, reversed_symmetric_point in zip(points, reversed_points, symmetric_points, reversed_symmetric_points):
    print(f"Tọa độ gốc: {original} => Tọa độ ngược: {reversed_point}")
    print(f"Tọa độ đối xứng: {symmetric_point} => Tọa độ ngược đối xứng: {reversed_symmetric_point}")
