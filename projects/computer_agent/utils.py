import numpy as np
import cv2

def create_white_canvas_with_red_dot(path):
    image = np.ones((1080, 1920, 3), dtype=np.uint8) * 255
    
    # Randomly choose the top-left corner for the 100x100 red dot
    x = np.random.randint(0, 1920 - 50)
    y = np.random.randint(0, 1080 - 50)

    # Set the area for the red dot, red in BGR is (0, 0, 255)
    image[y:y+50, x:x+50] = (0, 0, 255)
    cv2.imwrite(path, image)