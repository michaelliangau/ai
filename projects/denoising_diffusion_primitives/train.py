import datasets
import IPython
from PIL import Image
import torchvision
import model

# Hyperparameters

# Data
ds = datasets.load_dataset('HuggingFaceM4/COCO', '2014_captions', split='test') # TODO: Don't use the test split

# transform = torchvision.transforms.ToTensor()
# tensor_image = transform(ds[0]["image"])
# ds[0]["image"].save("original_image.jpg")

# Data Loader


# Forward Noising Step
# TODO: Build the Forward Noisng Step


# Backward Generation Step