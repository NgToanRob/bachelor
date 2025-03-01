import cv2
import numpy as np

def detect_lane(cv_image):
    gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.resize(gray_image, (640, 480))

    # Apply Gaussian Blur
    low_frequencies = cv2.GaussianBlur(gray_image, (9, 9), 0)

    # High-Pass Filter
    high_pass = cv2.subtract(gray_image, low_frequencies)

    # Sharpen Image
    sharp_highpass = cv2.addWeighted(gray_image, 1.5, high_pass, -0.5, 0)

    # Binary Thresholding
    thresh_value = 220
    ret, binary = cv2.threshold(sharp_highpass, thresh_value, 255, cv2.THRESH_BINARY)

    # Remove inside polygons
    mask = np.zeros_like(binary)
    pts = np.array([[260, 190], [380, 190], [415, 380], [225, 380]], np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)
    mask_inv = cv2.bitwise_not(mask)
    binary = cv2.bitwise_and(binary, binary, mask=mask_inv)

    # Crop down y < 190 and y > 300
    binary = binary[190:300, :]

    # Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    binary_clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_OPEN, kernel)

    # Find and draw contours
    contours, _ = cv2.findContours(binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    mask = np.zeros_like(binary_clean)
    cv2.drawContours(mask, contours, -1, 255, -1)
    binary_clean = cv2.bitwise_and(binary_clean, binary_clean, mask=mask)

    # For each row, loop from left to right, set 255 until meet 255, loop from right to left, set 255 until meet 255
    rows, cols = binary_clean.shape
    # left_done = False
    # right_done = False
    # for i in range(0, rows):
    #     col = 0
    #     if binary_clean[i, col] == 255:
    #         left_done = True
    #     while col < cols and binary_clean[i, col] == 0 and not left_done:
    #         binary_clean[i, col] = 255
    #         col += 1

    #     col = cols - 1
    #     if binary_clean[i, col] == 255:
    #         right_done = True
    #     while col >= 0 and binary_clean[i, col] == 0 and not right_done:
    #         binary_clean[i, col] = 255
    #         col -= 1

    # Apply Canny edge detection
    edges = cv2.Canny(binary_clean, threshold1=50, threshold2=190)


    # Hough Transform to extract lines
    rho = 2                # Độ phân giải khoảng cách (pixel)
    theta = np.pi / 190    # Độ phân giải góc (radians)
    hough_threshold = 15   # Số phiếu tối thiểu để xác định một đường thẳng
    min_line_length = 10   # Độ dài tối thiểu của đường thẳng
    max_line_gap = 5       # Khoảng cách nối liền các đoạn đường

    lines = cv2.HoughLinesP(edges, rho, theta, hough_threshold, 
                            minLineLength=min_line_length, maxLineGap=max_line_gap)

    # Draw lines on the original image
    line_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Skip horizontal lines
            if abs(y2 - y1) < 10:
                continue
            cv2.line(line_image, (x1, y1 + 190), (x2, y2 + 190), (0, 0, 255), 2)

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
        y2 = binary_clean.shape[0]
        left_line = extrapolate_line(left_slope, left_intercept, y1, y2)
        right_line = extrapolate_line(right_slope, right_intercept, y1, y2)

        # Draw the extrapolated lines on the image
        if left_line is not None:
            cv2.line(line_image, (left_line[0], left_line[1]+190), (left_line[2], left_line[3]+190), (255, 0, 0), 2)
        if right_line is not None:
            cv2.line(line_image, (right_line[0], right_line[1]+190), (right_line[2], right_line[3]+190), (255, 0, 0), 2)

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
        cv2.circle(line_image, (left_line_avg[0], left_line_avg[1]+190), 5, (0, 255, 0), -1)
        cv2.line(line_image, (left_line_avg[0], left_line_avg[1]+190), (left_line_avg[2], left_line_avg[3]+190), (0, 255, 255), 2)
    if right_line_avg is not None:
        cv2.circle(line_image, (right_line_avg[0], right_line_avg[1]+190), 5, (0, 255, 0), -1)
        cv2.line(line_image, (right_line_avg[0], right_line_avg[1]+190), (right_line_avg[2], right_line_avg[3]+190), (0, 255, 255), 2)

    # Ensure left_line_avg and right_line_avg are not None before returning
    if left_line_avg is not None and right_line_avg is not None:
        return line_image, ((left_line_avg[0], left_line_avg[1]+190), (right_line_avg[0], right_line_avg[1]+190))
    elif left_line_avg is None and right_line_avg is not None:
        return line_image, ((left_line_avg[0], left_line_avg[1]+190), (right_line_avg[0], right_line_avg[1]+190))
    elif left_line_avg is not None and right_line_avg is None:
        return line_image, ((left_line_avg[0], left_line_avg[1]+190), (left_line_avg[0], left_line_avg[1]+190))
    else:
        return line_image, None