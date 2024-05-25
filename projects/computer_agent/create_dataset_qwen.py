import numpy as np
import cv2
from tqdm import tqdm
import os
import json

NUM_IMAGES = 1000
DATASET_NAME = "2_dot_qwen_eval"

def create_images_with_red_dots(dataset_name: str, num_images: int = 1000):
    data = []
    os.makedirs(f'data/{dataset_name}', exist_ok=True)
    
    for idx in tqdm(range(num_images)):
        # Create a blank 1920x1080 image, RGB channels, white by default
        image = np.ones((1080, 1920, 3), dtype=np.uint8) * 255

        # Randomly choose the top-left corner for the 50x50 red dot
        x_red = np.random.randint(0, 1920 - 50)
        y_red = np.random.randint(0, 1080 - 50)
        image[y_red:y_red+50, x_red:x_red+50] = (0, 0, 255)  # Red dot

        # Randomly choose the top-left corner for the 50x50 green dot
        x_green = np.random.randint(0, 1920 - 50)
        y_green = np.random.randint(0, 1080 - 50)
        while x_green >= x_red - 50 and x_green <= x_red + 50 and y_green >= y_red - 50 and y_green <= y_red + 50:
            x_green = np.random.randint(0, 1920 - 50)
            y_green = np.random.randint(0, 1080 - 50)
        image[y_green:y_green+50, x_green:x_green+50] = (0, 255, 0)  # Green dot

        # Save the image
        image_path = f'/home/michael/ai/projects/computer_agent/data/{dataset_name}/{idx}.png'
        cv2.imwrite(image_path, image)

        # Construct JSON entry
        if idx % 2 == 0:
            target_color = "red"
            target_x = x_red + 25
            target_y = y_red + 25
        else:
            target_color = "green"
            target_x = x_green + 25
            target_y = y_green + 25

        entry = {
            "id": f"identity_{idx}",
            "conversations": [
                {
                    "from": "user",
                    "value": f"Picture 1: <img>{image_path}</img>\nClick on the {target_color} square."
                },
                {
                    "from": "assistant",
                    "value": f"<box>({target_x},{target_y})</box>"
                }
            ]
        }
        data.append(entry)

    # Save data to a JSON file
    with open(f'data/{dataset_name}.json', 'w') as f:
        json.dump(data, f, indent=4)

create_images_with_red_dots(dataset_name=DATASET_NAME, num_images=NUM_IMAGES)
