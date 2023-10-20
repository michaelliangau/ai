import torch
import transformers
import model
import utils
import tqdm

import sys
sys.path.append("../..")
import common.utils as common_utils

# Inputs
model_checkpoint = "./outputs/checkpoint_1.pt"

# Device
torch_device = common_utils.get_device("cpu")

# Text Embedding
tokenizer = transformers.T5TokenizerFast.from_pretrained("t5-small")
text_embedding_model = transformers.T5EncoderModel.from_pretrained("t5-small").to(torch_device)

# Initialize model and load checkpoint
unet = model.UNet().to(torch_device)
unet.load_state_dict(torch.load(model_checkpoint, map_location=torch_device)["model_state_dict"])
backward_process = model.BackwardProcess(model=unet, torch_device=torch_device)

# Get pure Gaussian noise image
noised_image = torch.randn((1, 3, 480, 640))  # 480 by 640 RGB image of pure Gaussian noise

# Get text prompt
text = "a man riding a red motorcycle"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(torch_device)
with torch.no_grad():
    outputs = text_embedding_model(**inputs)
text_embedding = outputs.last_hidden_state
mean_text_embedding = text_embedding.mean(dim=1)    

# Denoise image
for i in tqdm.tqdm(range(100)):
    with torch.no_grad():
        predicted_noise = backward_process.predict(image=noised_image, text=mean_text_embedding)
    noised_image = noised_image - predicted_noise
    if i % 10 == 0:
        utils.save_image(noised_image, f"./outputs/image_{i}.png")
utils.save_image(noised_image, "./outputs/image_final.png")
