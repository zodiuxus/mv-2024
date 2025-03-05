import numpy as np
import cv2
from math import floor, ceil

def local_binary_pattern(image, radius, num_points):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    lbp_image = np.zeros_like(image, dtype=np.uint8)
    
    height, width = image.shape
    
    angles = 2 * np.pi / num_points
    x_offsets = radius * np.cos(np.arange(num_points) * angles)
    y_offsets = -radius * np.sin(np.arange(num_points) * angles)
    
    x_offsets = np.round(x_offsets).astype(int)
    y_offsets = np.round(y_offsets).astype(int)
    
    for y in range(radius, height - radius):
        for x in range(radius, width - radius):
            center = image[y, x]
            binary_pattern = 0
            
            for i in range(num_points):
                nx = x + x_offsets[i]
                ny = y + y_offsets[i]
                
                if image[ny, nx] >= center:
                    binary_pattern |= (1 << (num_points - 1 - i))
            
            lbp_image[y, x] = binary_pattern
    
    return lbp_image

# Example usage
if __name__ == "__main__":
    image = cv2.imread("lab3/rat.png")
    
    radius = 1
    num_points = 8
    
    lbp_image = local_binary_pattern(image, radius, num_points)
    
    cv2.imshow("Original Image", image)
    cv2.imwrite("lbp_output.jpg", lbp_image)
    cv2.imshow("LBP Image", lbp_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()