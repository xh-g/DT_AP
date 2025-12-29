import cv2
import numpy as np

def create_sample_tangram():
    # Create a black background
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    
    # Draw a red triangle
    pts1 = np.array([[100, 100], [200, 100], [150, 200]], np.int32)
    cv2.fillPoly(img, [pts1], (0, 0, 255))
    
    # Draw a green square
    pts2 = np.array([[250, 100], [350, 100], [350, 200], [250, 200]], np.int32)
    cv2.fillPoly(img, [pts2], (0, 255, 0))
    
    # Draw a blue rotated rectangle
    rect = ((300, 350), (100, 50), 45)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    cv2.fillPoly(img, [box], (255, 0, 0))
    
    cv2.imwrite("tangram_sample.png", img)
    print("Created sample tangram_sample.png")

if __name__ == "__main__":
    create_sample_tangram()
