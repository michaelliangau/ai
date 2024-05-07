import numpy as np
import cv2
from tqdm import tqdm
import os
from datasets import Dataset

NUM_IMAGES=10000
DATASET_NAME="2_dot"

def create_images_with_red_dots(dataset_name: str, num_images: int = 1000):
    dataset_dict = {'image': [], 'label': [], 'text': []}
    os.makedirs(f'data/{dataset_name}', exist_ok=True)
    
    for idx in tqdm(range(num_images)):
        # Create a blank 1920x1080 image, RGB channels, white by default
        image = np.ones((1080, 1920, 3), dtype=np.uint8) * 255

        # Randomly choose the top-left corner for the 100x100 red dot
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

        # Save the image
        image_path = f'data/{dataset_name}/{idx}.png'
        cv2.imwrite(image_path, image)

        # Save the path and the label (the mid point of the dot)
        dataset_dict['image'].append(image_path)
        if idx % 2 == 0:
            dataset_dict['text'].append("Click on the red square.")
            dataset_dict['label'].append((x_red + 25, y_red + 25))
        else:
            dataset_dict['text'].append("Click on the green square.")
            dataset_dict['label'].append((x_green + 25, y_green + 25))
    # Create a Hugging Face dataset
    dataset = Dataset.from_dict(dataset_dict)
    dataset.save_to_disk('data/2_dot_dataset')

create_images_with_red_dots(dataset_name=DATASET_NAME, num_images=NUM_IMAGES)