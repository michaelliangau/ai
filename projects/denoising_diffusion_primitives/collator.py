import torchvision
import torch
import random

class Collator():
    """A class used to collate batches of data."""
    def __init__(self):
        """Initialize the Collator class with a transform that resizes images and converts them to tensors."""
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((480, 640)), # Resize images to 480 height and 640 width
            torchvision.transforms.Lambda(lambda x: x.convert('RGB')), # Convert images to 3 channels (RGB)
            torchvision.transforms.ToTensor() # Convert images to tensors
        ])    

    def collate(self, batch):
        """Collate a batch of data by transforming images and selecting a random sentence from each item.

        Args:
            batch (list): A list of items, each containing an image and sentences.

        Returns:
            dict: A dictionary containing transformed images and a list of randomly selected sentences.
        """
        images = [self.transform(item['image']) for item in batch]
        images = torch.stack(images, dim=0)
        collated = {
            "image": images,
            "sentences_raw": [random.choice(item['sentences_raw']) for item in batch]
        }
        return collated