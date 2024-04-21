from transformers import AutoImageProcessor, YolosModel
import torch
from datasets import load_dataset

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-small")
model = YolosModel.from_pretrained("hustvl/yolos-small")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
list(last_hidden_states.shape)
import IPython; IPython.embed()