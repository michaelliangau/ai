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

def create_white_canvas_with_2_dot(path):
    image = np.ones((1080, 1920, 3), dtype=np.uint8) * 255
    
    # Randomly choose the top-left corner for the 50x50 red dot
    x_red = np.random.randint(0, 1920 - 50)
    y_red = np.random.randint(0, 1080 - 50)

    # Set the area for the red dot, red in BGR is (0, 0, 255)
    image[y_red:y_red+50, x_red:x_red+50] = (0, 0, 255)

    # Randomly choose the top-left corner for the 50x50 green dot
    x_green = np.random.randint(0, 1920 - 50)
    y_green = np.random.randint(0, 1080 - 50)

    # Ensure the green dot does not overlap with the red dot
    while x_green >= x_red - 50 and x_green <= x_red + 50 and y_green >= y_red - 50 and y_green <= y_red + 50:
        x_green = np.random.randint(0, 1920 - 50)
        y_green = np.random.randint(0, 1080 - 50)

    # Set the area for the green dot, green in BGR is (0, 255, 0)
    image[y_green:y_green+50, x_green:x_green+50] = (0, 255, 0)

    cv2.imwrite(path, image)