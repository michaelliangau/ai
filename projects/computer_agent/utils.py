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

def resize_to_max(image, max_width=1920, max_height=1080):
    width, height = image.size
    if width <= max_width and height <= max_height:
        return image

    scale = min(max_width/width, max_height/height)
    width = int(width*scale)
    height = int(height*scale)

    return image.resize((width, height), Image.LANCZOS)

def pad_to_size(image, canvas_width=1920, canvas_height=1080):
    width, height = image.size
    if width >= canvas_width and height >= canvas_height:
        return image

    # Paste at (0, 0)
    canvas = Image.new("RGB", (canvas_width, canvas_height))
    canvas.paste(image)
    return canvas

def get_middle_of_rect(x: int, y: int, height: int, width: int):
    """
    Calculate the middle coordinates of a rectangle given its top-left corner,
    height, and width.

    Parameters:
        x (int): The x-coordinate of the top-left corner of the rectangle.
        y (int): The y-coordinate of the top-left corner of the rectangle.
        height (int): The height of the rectangle.
        width (int): The width of the rectangle.

    Returns:
        tuple: A tuple containing the middle x and y coordinates (middle_x, middle_y).
    """

    middle_x = x + width // 2
    middle_y = y + height // 2
    return middle_x, middle_y
