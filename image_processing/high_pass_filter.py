import cv2
import numpy as np

# Đọc ảnh và chuyển sang xám (nếu ảnh gốc là màu)
image = cv2.imread("/home/toan/Pictures/Screenshots/Screenshot from 2025-03-01 22-26-49.png", cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError("Không tìm thấy ảnh, vui lòng kiểm tra đường dẫn.")

# Scale ảnh về kích thước (640, 480)
image = cv2.resize(image, (640, 480))

# Hiển thị ảnh gốc
cv2.imshow("Original (Grayscale)", image)

# Tạo ảnh mờ (Low Frequencies) bằng Gaussian Blur
low_frequencies = cv2.GaussianBlur(image, (9, 9), 0)
cv2.imshow("Low Frequencies (Blurred)", low_frequencies)

# Trừ ảnh mờ khỏi ảnh gốc để lấy phần tần số cao (High-Pass)
high_pass = cv2.subtract(image, low_frequencies)
cv2.imshow("High-Pass", high_pass)

# Tăng cường độ sắc nét bằng cách kết hợp ảnh gốc với high_pass
sharp_highpass = cv2.addWeighted(image, 1.5, high_pass, -0.5, 0)
cv2.imshow("Sharpened (High-Pass Filtered)", sharp_highpass)

# Adaptive Thresholding
block_size = 11
C = 2
binary = cv2.adaptiveThreshold(sharp_highpass, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C)
cv2.imshow("Adaptive Threshold", binary)

# Gaussian Blur
binary = cv2.GaussianBlur(binary, (7, 7), 0)
cv2.imshow("Gaussian Blur", binary)



# crop down y < 150 and y > 390
binary = binary[150:390, :]
cv2.imshow("Binary Threshold (Cropped)", binary)

# Remove the upper left and right triangles with top size if 150 and all y
rows, cols = binary.shape
for i in range(0, rows):
    for j in range(0, cols):
        if j < ((rows-i)/rows*150) or j > 640 - (rows-i)/rows*150:
            binary[i, j] = 0
            
cv2.imshow("Binary Threshold (After Mask)", binary)


# adaptive thresholding
block_size = 11
C = 2
binary = cv2.adaptiveThreshold(binary, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C)
cv2.imshow("Adaptive Threshold 2", binary)


# Remove inside polygons
mask = np.zeros_like(binary)
pts = np.array([[280, 190-150], [360, 190-150], [415, 380-150], [225, 380-150]], np.int32).reshape((-1, 1, 2))
cv2.fillPoly(mask, [pts], 255)
mask_inv = cv2.bitwise_not(mask)
binary = cv2.bitwise_and(binary, binary, mask=mask_inv)
cv2.imshow("Binary Threshold (After Mask)", binary)

# reverse the binary image
binary = cv2.bitwise_not(binary)
cv2.imshow("Binary Threshold (Reversed)", binary)

# -------------------------------------------------
# Tích hợp Hough Transform để trích xuất các đường thẳng
# -------------------------------------------------
# Sử dụng HoughLinesP để phát hiện các đường thẳng từ ảnh edge
rho = 5                # Độ phân giải khoảng cách (pixel)
theta = np.pi / 180    # Độ phân giải góc (radians)
hough_threshold = 15   # Số phiếu tối thiểu để xác định một đường thẳng
min_line_length = 50   # Độ dài tối thiểu của đường thẳng
max_line_gap = 5       # Khoảng cách nối liền các đoạn đường

lines = cv2.HoughLinesP(binary, rho, theta, hough_threshold, 
                        minLineLength=min_line_length, maxLineGap=max_line_gap)

line_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
# Remove lines that tend to be horizontal
if lines is not None:
    lines = [line for line in lines if abs(line[0][3] - line[0][1]) > 70]

polygon = np.array([[280, 190-150], [360, 190-150], [415, 380-150], [225, 380-150]])
# create polygon contour
polygon_contour = np.array([(x, y) for x, y in polygon], np.int32)
# Hàm kiểm tra giao nhau của 2 đoạn thẳng sử dụng thuật toán ccw
def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def segments_intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

# Hàm kiểm tra một đoạn đường thẳng có giao với bất kỳ cạnh nào của polygon hay không
def line_intersects_polygon(x1, y1, x2, y2, polygon):
    pts = polygon.reshape(-1, 2)
    A = (x1, y1)
    B = (x2, y2)
    n = len(pts)
    for i in range(n):
        C = tuple(pts[i])
        D = tuple(pts[(i+1) % n])
        if segments_intersect(A, B, C, D):
            return True
    return False

# Loại bỏ các đường thẳng nằm trong hoặc đi qua polygon
print(len(polygon_contour))
filtered_lines = []
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Convert to Python integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # Sử dụng cv2.pointPolygonTest: trả về >0 nếu nằm trong, =0 nếu nằm trên biên, <0 nếu nằm ngoài
        d1 = cv2.pointPolygonTest(polygon_contour, (x1, y1), False)
        d2 = cv2.pointPolygonTest(polygon_contour, (x2, y2), False)
        
        # Nếu một trong hai điểm nằm trong (>= 0) thì loại bỏ đường thẳng
        if d1 >= 0 or d2 >= 0:
            continue
        
        # Nếu cả 2 điểm nằm ngoài nhưng đường thẳng cắt qua biên polygon thì loại bỏ
        if line_intersects_polygon(x1, y1, x2, y2, polygon_contour):
            continue
        
        filtered_lines.append(line)
    
    lines = filtered_lines

    # Function to calculate the slope and intercept of a line
    def calculate_slope_intercept(x1, y1, x2, y2):
        if x2 - x1 == 0:
            return None, None
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        return slope, intercept
    
    # Divide lines into left and right groups based on slope
    left_lines = []
    right_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope, intercept = calculate_slope_intercept(x1, y1, x2, y2)
        if slope is not None:
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            weight = length ** 2
            if slope < 0 and x1 < cols / 2 and x2 < cols / 2:
                left_lines.append((slope, intercept, weight))
            elif slope > 0 and x1 > cols / 2 and x2 > cols / 2:
                right_lines.append((slope, intercept, weight))
    
    print('Left lines:', len(left_lines))
    print('Right lines:', len(right_lines))

    # Function to calculate weighted average of slopes and intercepts
    def weighted_average(lines):
        if not lines:
            return None, None
        slope_sum = sum(slope * weight for slope, intercept, weight in lines)
        intercept_sum = sum(intercept * weight for slope, intercept, weight in lines)
        weight_sum = sum(weight for slope, intercept, weight in lines)
        return slope_sum / weight_sum, intercept_sum / weight_sum

    # Calculate weighted average for left and right lines
    left_slope, left_intercept = weighted_average(left_lines)
    right_slope, right_intercept = weighted_average(right_lines)

    # Function to extrapolate lines
    def extrapolate_line(slope, intercept, y1, y2):
        if slope is None or intercept is None or slope == 0:
            return None
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return x1, y1, x2, y2

    # Extrapolate left and right lines
    y1 = 0
    y2 = binary.shape[0]
    left_line = extrapolate_line(left_slope, left_intercept, y1, y2)
    right_line = extrapolate_line(right_slope, right_intercept, y1, y2)

    # Draw the extrapolated lines on the image
    if left_line is not None:
        cv2.line(line_image, (left_line[0], left_line[1]+150), (left_line[2], left_line[3]+150), (255, 0, 0), 2)
    if right_line is not None:
        cv2.line(line_image, (right_line[0], right_line[1]+150), (right_line[2], right_line[3]+150), (255, 0, 0), 2)

    
# Define the polygon to exclude
line_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_image, (x1, y1+150), (x2, y2+150), (0, 0, 255), 2)
cv2.imshow("Hough Lines", line_image)
    

# Initialize variables to store cumulative moving average of line parameters
n = 20  # Number of frames to average over
left_slopes = []
left_intercepts = []
right_slopes = []
right_intercepts = []

# Function to calculate cumulative moving average
def cumulative_moving_average(values, new_value, n):
    values.append(new_value)
    if len(values) > n:
        values.pop(0)
    return sum(values) / len(values)

# Update cumulative moving average for left and right lines
if left_slope is not None and left_intercept is not None:
    left_slope_avg = cumulative_moving_average(left_slopes, left_slope, n)
    left_intercept_avg = cumulative_moving_average(left_intercepts, left_intercept, n)
else:
    left_slope_avg = np.mean(left_slopes) if left_slopes else None
    left_intercept_avg = np.mean(left_intercepts) if left_intercepts else None

if right_slope is not None and right_intercept is not None:
    right_slope_avg = cumulative_moving_average(right_slopes, right_slope, n)
    right_intercept_avg = cumulative_moving_average(right_intercepts, right_intercept, n)
else:
    right_slope_avg = np.mean(right_slopes) if right_slopes else None
    right_intercept_avg = np.mean(right_intercepts) if right_intercepts else None

# Extrapolate lines using the averaged parameters
left_line_avg = extrapolate_line(left_slope_avg, left_intercept_avg, y1, y2)
right_line_avg = extrapolate_line(right_slope_avg, right_intercept_avg, y1, y2)

# Draw the averaged extrapolated lines on the image
if left_line_avg is not None:
    cv2.circle(line_image, (left_line_avg[0], left_line_avg[1]+150), 5, (0, 255, 0), -1)
    cv2.line(line_image, (left_line_avg[0], left_line_avg[1]+150), (left_line_avg[2], left_line_avg[3]+150), (0, 255, 255), 2)
if right_line_avg is not None:
    cv2.circle(line_image, (right_line_avg[0], right_line_avg[1]+150), 5, (0, 255, 0), -1)
    cv2.line(line_image, (right_line_avg[0], right_line_avg[1]+150), (right_line_avg[2], right_line_avg[3]+150), (0, 255, 255), 2)

cv2.imshow("Averaged Hough Lines (Cumulative Moving Average)", line_image)

# # Remove noise
# kernel = np.ones((3, 3), np.uint8)
# binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
# binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
# cv2.imshow("Binary Threshold (After Morphological Operations)", binary)

# # apply sobel
# # sobel_x = cv2.Sobel(binary, cv2.CV_64F, 1, 0, ksize=3)
# # sobel_y = cv2.Sobel(binary, cv2.CV_64F, 0, 1, ksize=3)
# # sobel = cv2.magnitude(sobel_x, sobel_y)
# # cv2.imshow("Sobel", sobel)

# # apply Bộ lọc Prewitt theo hướng dọc
# kernel = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
# prewitt_x = cv2.filter2D(binary, -1, kernel)
# cv2.imshow("Prewitt X", prewitt_x) 


# # Apply Canny edge detection
# edges = cv2.Canny(binary, threshold1=50, threshold2=150)
# cv2.imshow("Edges (Canny)", edges)

# # Morphological operations để xóa nhiễu hoặc lấp lỗ hổng
# kernel = np.ones((3,3), np.uint8)
# binary_clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
# binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_OPEN, kernel)
# cv2.imshow("Binary Clean", binary_clean)

# Remove inside polygons
# mask = np.zeros_like(binary)
# pts = np.array([[260, 190], [380, 190], [415, 380], [225, 380]], np.int32).reshape((-1, 1, 2))
# cv2.fillPoly(mask, [pts], 255)
# mask_inv = cv2.bitwise_not(mask)
# binary = cv2.bitwise_and(binary, binary, mask=mask_inv)
# cv2.imshow("Binary Threshold (After Mask)", binary)



# # Apply morphological opening to remove noise
# kernel = np.ones((5, 5), np.uint8)
# binary_opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
# cv2.imshow("Binary Opened", binary_opened)


# # Apply Erosion to remove noise
# kernel = np.ones((5, 5), np.uint8)
# binary_eroded = cv2.erode(binary, kernel, iterations=1)
# cv2.imshow("Binary Eroded", binary_eroded)

# # Apply Dilation to enhance features
# binary_dilated = cv2.dilate(binary_eroded, kernel, iterations=1)
# cv2.imshow("Binary Dilated", binary_dilated)





# ROI 

# # Tìm và vẽ các đường biên (contours)
# contours, _ = cv2.findContours(binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# contour_image = cv2.cvtColor(binary_clean, cv2.COLOR_GRAY2BGR)
# cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
# cv2.imshow("Contours", contour_image)

# # Lấy 2 contours có diện tích lớn nhất và giữ lại
# contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
# mask = np.zeros_like(binary_clean)
# cv2.drawContours(mask, contours, -1, 255, -1)
# binary_clean = cv2.bitwise_and(binary_clean, binary_clean, mask=mask)
# cv2.imshow("Binary Clean (After Mask 3)", binary_clean)

# # for each row, loop from left to right, set 255 until meet 255, loop from right to left, set 255 until meet 255
# # rows, cols = binary_clean.shape
# # left_done = False
# # right_done = False
# # for i in range(0, rows):
# #     col = 0
# #     if binary_clean[i, col] == 255:
# #         left_done = True
# #     while col < cols and binary_clean[i, col] == 0 and not left_done:
# #         binary_clean[i, col] = 255
# #         col += 1

# #     col = cols - 1
# #     if binary_clean[i, col] == 255:
# #         right_done = True
# #     while col >= 0 and binary_clean[i, col] == 0 and not right_done:
# #         binary_clean[i, col] = 255
# #         col -= 1
        
# # cv2.imshow("Binary Clean (After remove outside)", binary_clean)


    














# # Function to calculate the slope and intercept of a line
# def calculate_slope_intercept(x1, y1, x2, y2):
#     if x2 - x1 == 0:
#         return None, None
#     slope = (y2 - y1) / (x2 - x1)
#     intercept = y1 - slope * x1
#     return slope, intercept

# # Divide lines into left and right groups based on slope
# left_lines = []
# right_lines = []
# for line in lines:
#     x1, y1, x2, y2 = line[0]
#     slope, intercept = calculate_slope_intercept(x1, y1, x2, y2)
#     if slope is not None:
#         length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
#         weight = length ** 2
#         if slope < 0:
#             left_lines.append((slope, intercept, weight))
#         else:
#             right_lines.append((slope, intercept, weight))

# # Function to calculate weighted average of slopes and intercepts
# def weighted_average(lines):
#     if not lines:
#         return None, None
#     slope_sum = sum(slope * weight for slope, intercept, weight in lines)
#     intercept_sum = sum(intercept * weight for slope, intercept, weight in lines)
#     weight_sum = sum(weight for slope, intercept, weight in lines)
#     return slope_sum / weight_sum, intercept_sum / weight_sum

# # Calculate weighted average for left and right lines
# left_slope, left_intercept = weighted_average(left_lines)
# right_slope, right_intercept = weighted_average(right_lines)

# # Function to extrapolate lines
# def extrapolate_line(slope, intercept, y1, y2):
#     if slope is None or intercept is None:
#         return None
#     x1 = int((y1 - intercept) / slope)
#     x2 = int((y2 - intercept) / slope)
#     return x1, y1, x2, y2

# # Extrapolate left and right lines
# y1 = 0
# y2 = binary_clean.shape[0]
# left_line = extrapolate_line(left_slope, left_intercept, y1, y2)
# right_line = extrapolate_line(right_slope, right_intercept, y1, y2)

# # Draw the extrapolated lines on the image
# if left_line is not None:
#     cv2.line(line_image, (left_line[0], left_line[1]+150), (left_line[2], left_line[3]+150), (255, 0, 0), 2)
# if right_line is not None:
#     cv2.line(line_image, (right_line[0], right_line[1]+150), (right_line[2], right_line[3]+150), (255, 0, 0), 2)
# cv2.imshow("Averaged Hough Lines", line_image)


# # Initialize variables to store cumulative moving average of line parameters
# n = 10  # Number of frames to average over
# left_slopes = []
# left_intercepts = []
# right_slopes = []
# right_intercepts = []

# # Function to calculate cumulative moving average
# def cumulative_moving_average(values, new_value, n):
#     values.append(new_value)
#     if len(values) > n:
#         values.pop(0)
#     return sum(values) / len(values)

# # Update cumulative moving average for left and right lines
# if left_slope is not None and left_intercept is not None:
#     left_slope_avg = cumulative_moving_average(left_slopes, left_slope, n)
#     left_intercept_avg = cumulative_moving_average(left_intercepts, left_intercept, n)
# else:
#     left_slope_avg = np.mean(left_slopes) if left_slopes else None
#     left_intercept_avg = np.mean(left_intercepts) if left_intercepts else None

# if right_slope is not None and right_intercept is not None:
#     right_slope_avg = cumulative_moving_average(right_slopes, right_slope, n)
#     right_intercept_avg = cumulative_moving_average(right_intercepts, right_intercept, n)
# else:
#     right_slope_avg = np.mean(right_slopes) if right_slopes else None
#     right_intercept_avg = np.mean(right_intercepts) if right_intercepts else None

# # Extrapolate lines using the averaged parameters
# left_line_avg = extrapolate_line(left_slope_avg, left_intercept_avg, y1, y2)
# right_line_avg = extrapolate_line(right_slope_avg, right_intercept_avg, y1, y2)

# # Draw the averaged extrapolated lines on the image
# if left_line_avg is not None:
#     cv2.line(line_image, (left_line_avg[0], left_line_avg[1]+150), (left_line_avg[2], left_line_avg[3]+150), (0, 255, 255), 2)
# if right_line_avg is not None:
#     cv2.line(line_image, (right_line_avg[0], right_line_avg[1]+150), (right_line_avg[2], right_line_avg[3]+150), (0, 255, 255), 2)
# cv2.imshow("Averaged Hough Lines (Cumulative Moving Average)", line_image)

# Chờ phím và đóng tất cả cửa sổ
cv2.waitKey(0)
cv2.destroyAllWindows()
