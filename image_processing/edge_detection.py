import cv2
import numpy as np

# 1. Load the original image
frame = cv2.imread('/home/toan/Pictures/Screenshots/Screenshot from 2025-02-28 20-00-17.png')  # Replace with your actual image path
if frame is None:
    raise FileNotFoundError("Could not load image. Check the file path.")

# 2. Show the original image
cv2.imshow("Original Image", frame)

# 3. Convert to grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale", gray)

alpha = 1.5
beta = 50
gray_contrast = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
cv2.imshow("Grayscale + Increased Contrast (convertScaleAbs)", gray_contrast)

# gray_eq = cv2.equalizeHist(gray)
# cv2.imshow("Grayscale + Equalized (Histogram Equalization)", gray_eq)


# 4. Reduce noise (Gaussian Blur)
blur = cv2.GaussianBlur(gray_contrast, (3, 3), 0)
cv2.imshow("Blurred (Noise Reduced)", blur)

# 5. Edge detection (Canny)
#    - You can adjust threshold1 and threshold2 to tune sensitivity
edges = cv2.Canny(blur, threshold1=50, threshold2=150)
cv2.imshow("Edges (Canny)", edges)

# 6. Wait for a key press and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()



