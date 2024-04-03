from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import browser
import torch
import constants
torch.manual_seed(1234)

# Define vars
URL = "https://docs.google.com/forms/d/e/1FAIpQLSeSxglhKz5qludFOp4w3diD58RXFJbB-cXVeuE3PaXTkmnEGg/viewform"

# Browser
print("Creating browser instance")
sandbox = browser.Browser()
print("Opening URL")
sandbox.open_url(url=URL)
sandbox.take_screenshot(screenshot_path=constants.DEFAULT_STATE_SCREENSHOT_PATH)
print("Screenshot taken")


# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

# use cuda device
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True, bf16=True).eval()

# Specify hyperparameters for generation
model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

# Out[2]: <PIL.PngImagePlugin.PngImageFile image mode=RGBA size=1920x1080>
# 1st dialogue turn
query = tokenizer.from_list_format([
    {'image': 'tmp/state.png'}, # Either a local path or an url
    {'text': 'Give me a step by step plan on how I can submit this form. Tell me exactly what to click and type.'},
])
response, history = model.chat(tokenizer, query=query, history=None)

# 2nd dialogue turn TODO: Can't seem to make this work consistently with English.
response, history = model.chat(tokenizer, '框出图中名字收入字段的位置', history=history)

image = tokenizer.draw_bbox_on_latest_picture(response, history)
if image:
  image.save('data/1.jpg')
else:
  print("no box")
exit()


# Map natural language actions to UI actions