
from transformers import FuyuForCausalLM, BitsAndBytesConfig, FuyuProcessor
from PIL import Image, ImageDraw
import utils
import browser
import torch
import constants

# Define vars
URL = "https://docs.google.com/forms/d/e/1FAIpQLSeSxglhKz5qludFOp4w3diD58RXFJbB-cXVeuE3PaXTkmnEGg/viewform"

# Browser
print("Creating browser instance")
sandbox = browser.Browser()
print("Opening URL")
sandbox.open_url(url=URL)
sandbox.take_screenshot(screenshot_path=constants.DEFAULT_STATE_SCREENSHOT_PATH)
print("Screenshot taken")

# Multimodal model
pretrained_path = "adept/fuyu-8b"
processor = FuyuProcessor.from_pretrained(pretrained_path)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)
model = FuyuForCausalLM.from_pretrained(pretrained_path, quantization_config=quantization_config, low_cpu_mem_usage=True)
model.eval()

# Create the plan using RCI prompting
prompt = f"You are Anton, a computer agent tasked with teaching your son how to use a computer. Your goal is perform the task on the screen as accurately as possible. You only have access to 3 actions:\n1. Move the mouse\n2. Click the mouse\n3. Type something\n Come up with a step by step plan to complete the form for your son.\n"
image_pil = Image.open(constants.DEFAULT_STATE_SCREENSHOT_PATH).convert("RGB")
image_pil = utils.resize_to_max(image_pil)
image_pil = utils.pad_to_size(image_pil)
model_inputs = processor(text=prompt, images=[image_pil]).to('cuda')
outputs = model.generate(**model_inputs, max_new_tokens=500)
text = processor.batch_decode(outputs[:, -500:], skip_special_tokens=True)
print(text)

import IPython; IPython.embed()





bbox_prompt = f"When presented with a box, perform OCR to extract text contained within it. If provided with text, generate the corresponding bounding box.\nComments"
bbox_image_pil = Image.open("data/1920_1080.png").convert("RGB")

padded = utils.resize_to_max(bbox_image_pil)
padded = utils.pad_to_size(padded)

model_inputs = processor(text=bbox_prompt, images=[padded]).to('cuda')

outputs = model.generate(**model_inputs, max_new_tokens=40)

post_processed_bbox_tokens = processor.post_process_box_coordinates(outputs)[0]
decoded = processor.decode(post_processed_bbox_tokens, skip_special_tokens=True)
decoded = decoded.split('\x04', 1)[1] if '\x04' in decoded else ''
coords = utils.coords_from_response(decoded)

# TODO: Delete. DEBUG: Draw bbox on image and save
utils.draw_bbox_and_save(padded, coords, "data/x_bbox.png")

# Map natural language actions to UI actions