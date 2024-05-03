import numpy as np
import cv2
from tqdm import tqdm
import os
from datasets import Dataset

def create_images_with_red_dots(num_images=1000):
    dataset_dict = {'image': [], 'label': [], 'text': []}
    os.makedirs('data/red_dot', exist_ok=True)
    
    for idx in tqdm(range(num_images)):
        # Create a blank 1920x1080 image, RGB channels, white by default
        image = np.ones((1080, 1920, 3), dtype=np.uint8) * 255

        # Randomly choose the top-left corner for the 100x100 red dot
        x = np.random.randint(0, 1920 - 50)
        y = np.random.randint(0, 1080 - 50)

        # Set the area for the red dot, red in BGR is (0, 0, 255)
        image[y:y+50, x:x+50] = (0, 0, 255)

        # Save the image
        image_path = f'data/red_dot/{idx}.png'
        cv2.imwrite(image_path, image)

        # Save the path and the label (the mid point of the red dot)
        dataset_dict['image'].append(image_path)
        dataset_dict['text'].append("Click on the red square.")
        dataset_dict['label'].append((x + 25, y + 25)) # Center of the square
    # Create a Hugging Face dataset
    dataset = Dataset.from_dict(dataset_dict)
    dataset.save_to_disk('data/red_dot_dataset')

create_images_with_red_dots()
