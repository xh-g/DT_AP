import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def get_orientation(contour):
    """
    Calculates the orientation of a contour using PCA.
    Returns the angle in degrees.
    """
    sz = len(contour)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = contour[i,0,0]
        data_pts[i,1] = contour[i,0,1]
    
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    
    angle = math.atan2(eigenvectors[0,1], eigenvectors[0,0]) # Orientation in radians
    return math.degrees(angle), (mean[0,0], mean[0,1])

def process_tangram(image_path):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection or Thresholding
    # Using Canny edge detection as there are gaps between pieces
    # Lower thresholds to detect darker colors in grayscale
    edges = cv2.Canny(blur, 30, 100)
    
    # Dilate edges to close small gaps if necessary, but user said there are gaps so we might want to keep them distinct
    # kernel = np.ones((3,3), np.uint8)
    # edges = cv2.dilate(edges, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Found {len(contours)} contours")

    output_img = img.copy()
    
    results = []

    for i, cnt in enumerate(contours):
        # Filter small contours (noise)
        area = cv2.contourArea(cnt)
        if area < 1000: # Adjust threshold based on image resolution
            continue
            
        # Approximate polygon
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        # Calculate orientation and centroid
        angle, center = get_orientation(cnt)
        cx, cy = int(center[0]), int(center[1])
        
        results.append({
            "id": i,
            "center": (cx, cy),
            "angle": angle,
            "vertices": len(approx)
        })
        
        # Draw contours and info
        cv2.drawContours(output_img, [cnt], -1, (0, 255, 0), 2)
        
        # Draw orientation axis
        length = 50
        x2 = int(cx + length * math.cos(math.radians(angle)))
        y2 = int(cy + length * math.sin(math.radians(angle)))
        cv2.line(output_img, (cx, cy), (x2, y2), (0, 0, 255), 2)
        
        # Put text
        cv2.putText(output_img, f"ID:{i}", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(output_img, f"Ang:{int(angle)}", (cx - 20, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        print(f"Piece {i}: Center=({cx}, {cy}), Angle={angle:.2f}, Vertices={len(approx)}")

    # Show result
    # cv2.imshow("Tangram Detection", output_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # Save result
    cv2.imwrite("result.png", output_img)
    print("Result saved to result.png")

if __name__ == "__main__":
    # Replace with actual image path
    # Using generated sample for demonstration as the original file was inaccessible
    process_tangram("image.png")
