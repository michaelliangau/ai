import numpy as np
import cv2
from tqdm import tqdm
import os
import json

NUM_IMAGES = 2
DATASET_NAME = "qwen_multistep"
BASE_DIR_PATH = "/Users/michael/Desktop/wip"

def create_images_with_red_dots(dataset_name: str, num_images: int = 1000):
    data = []
    os.makedirs(f'../data/{dataset_name}', exist_ok=True)
    
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

        # Randomly choose the top-left corner for the 50x50 blue dot
        x_blue = np.random.randint(0, 1920 - 50)
        y_blue = np.random.randint(0, 1080 - 50)
        while (x_blue >= x_red - 50 and x_blue <= x_red + 50 and y_blue >= y_red - 50 and y_blue <= y_red + 50) or \
              (x_blue >= x_green - 50 and x_blue <= x_green + 50 and y_blue >= y_green - 50 and y_blue <= y_green + 50):
            x_blue = np.random.randint(0, 1920 - 50)
            y_blue = np.random.randint(0, 1080 - 50)
        image[y_blue:y_blue+50, x_blue:x_blue+50] = (255, 0, 0)  # Blue dot

        # Save the initial image
        image_path = f'{BASE_DIR_PATH}/ai/projects/computer_agent/data/{dataset_name}/{idx}_initial.png'
        cv2.imwrite(image_path, image)

        # Choose a random color for the first click
        colors = ['red', 'green', 'blue']
        first_color = np.random.choice(colors)
        colors.remove(first_color)

        # Construct JSON entry for the first turn
        entry = {
            "id": f"identity_{idx}",
            "conversations": [
                {
                    "from": "user",
                    "value": f"Picture 1: <img>{image_path}</img>\nClick on the {first_color} square."
                },
                {
                    "from": "assistant",
                    "value": f"<box>({x_red + 25},{y_red + 25})</box>" if first_color == "red" else
                             f"<box>({x_green + 25},{y_green + 25})</box>" if first_color == "green" else
                             f"<box>({x_blue + 25},{y_blue + 25})</box>"
                }
            ]
        }

        # Remove the first color and save a new image
        if first_color == 'red':
            image[y_red:y_red+50, x_red:x_red+50] = (255, 255, 255)
        elif first_color == 'green':
            image[y_green:y_green+50, x_green:x_green+50] = (255, 255, 255)
        else:
            image[y_blue:y_blue+50, x_blue:x_blue+50] = (255, 255, 255)

        second_image_path = f'{BASE_DIR_PATH}/ai/projects/computer_agent/data/{dataset_name}/{idx}_second.png'
        cv2.imwrite(second_image_path, image)

        # Choose a random color for the second click
        second_color = np.random.choice(colors)
        colors.remove(second_color)

        # Add second turn to the conversation
        entry["conversations"].append({
            "from": "user",
            "value": f"Picture 2: <img>{second_image_path}</img>\nClick on the {second_color} square."
        })
        entry["conversations"].append({
            "from": "assistant",
            "value": f"<box>({x_red + 25},{y_red + 25})</box>" if second_color == "red" else
                     f"<box>({x_green + 25},{y_green + 25})</box>" if second_color == "green" else
                     f"<box>({x_blue + 25},{y_blue + 25})</box>"
        })

        # Remove the second color and save a new image
        if second_color == 'red':
            image[y_red:y_red+50, x_red:x_red+50] = (255, 255, 255)
        elif second_color == 'green':
            image[y_green:y_green+50, x_green:x_green+50] = (255, 255, 255)
        else:
            image[y_blue:y_blue+50, x_blue:x_blue+50] = (255, 255, 255)

        third_image_path = f'{BASE_DIR_PATH}/ai/projects/computer_agent/data/{dataset_name}/{idx}_third.png'
        cv2.imwrite(third_image_path, image)

        # The last color
        last_color = colors[0]

        # Add third turn to the conversation
        entry["conversations"].append({
            "from": "user",
            "value": f"Picture 3: <img>{third_image_path}</img>\nClick on the {last_color} square."
        })
        entry["conversations"].append({
            "from": "assistant",
            "value": f"<box>({x_red + 25},{y_red + 25})</box>" if last_color == "red" else
                     f"<box>({x_green + 25},{y_green + 25})</box>" if last_color == "green" else
                     f"<box>({x_blue + 25},{y_blue + 25})</box>"
        })

        data.append(entry)

    # Save data to a JSON file
    with open(f'../data/{dataset_name}.json', 'w') as f:
        json.dump(data, f, indent=4)

create_images_with_red_dots(dataset_name=DATASET_NAME, num_images=NUM_IMAGES)
