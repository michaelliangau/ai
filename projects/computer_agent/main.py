from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from transformers import FuyuForCausalLM, BitsAndBytesConfig, FuyuProcessor
from PIL import Image, ImageDraw
import utils
import torch

# URL you want to open
url = "https://docs.google.com/forms/d/e/1FAIpQLSeSxglhKz5qludFOp4w3diD58RXFJbB-cXVeuE3PaXTkmnEGg/viewform"

# Set up Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless")  # Optional: if you're running in a headless environment
chrome_options.add_argument("--no-sandbox")  # Bypass OS security model, REQUIRED on Linux if not root
chrome_options.add_argument("--window-size=1920,1080")

# Connect to the Selenium Server running in Docker
driver = webdriver.Remote(
    command_executor='http://localhost:4444/wd/hub',
    options=chrome_options  # Use the options argument instead of desired_capabilities
)

# Open the URL
driver.get(url)

# Specify the path and filename where you want to save the screenshot
screenshot_path = "data/screenshot.png"

# Take a screenshot of the current window and save it to the specified file
driver.save_screenshot(screenshot_path)

# Multimodal model
pretrained_path = "adept/fuyu-8b"
processor = FuyuProcessor.from_pretrained(pretrained_path)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = FuyuForCausalLM.from_pretrained(pretrained_path, quantization_config=quantization_config)
model.eval()

bbox_prompt = f"When presented with a box, perform OCR to extract text contained within it. If provided with text, generate the corresponding bounding box.\nComments"
bbox_image_pil = Image.open("data/1920_1080.png")
bbox_image_pil = bbox_image_pil.convert("RGB")

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