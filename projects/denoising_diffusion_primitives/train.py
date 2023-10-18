import datasets
import IPython
from PIL import Image
import torchvision
import model

# Hyperparameters
forward_beta = 0.2
forward_num_timesteps = 100
forward_decay_rate = 0.98

# Data
ds = datasets.load_dataset('HuggingFaceM4/COCO', '2014_captions', split='test') # TODO: Don't use the test split
import IPython; IPython.embed()
transform = torchvision.transforms.ToTensor()
image = transform(ds[0]["image"])

# Data Loader


# Forward Noising Step
forward_process = model.ForwardProcess(num_timesteps=forward_num_timesteps, initial_beta=forward_beta, decay_rate=forward_decay_rate)
noised_image = forward_process.sample(image, timestep=0)



# Backward Generation Step