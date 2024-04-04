# from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers.generation import GenerationConfig
# import torch
# import browser
# import torch
# import constants
# torch.manual_seed(1234)

# # Define vars
# URL = "https://docs.google.com/forms/d/e/1FAIpQLSeSxglhKz5qludFOp4w3diD58RXFJbB-cXVeuE3PaXTkmnEGg/viewform"

# # Browser
# print("Creating browser instance")
# sandbox = browser.Browser()
# print("Opening URL")
# sandbox.open_url(url=URL)
# sandbox.take_screenshot(screenshot_path=constants.DEFAULT_STATE_SCREENSHOT_PATH)
# print("Screenshot taken")


# # Note: The default behavior now has injection attack prevention off.
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True, bf16=True).eval()

# # Specify hyperparameters for generation
# model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

# # Initial prompt
# query = tokenizer.from_list_format([
#     {'image': 'tmp/state.png'}, # Either a local path or an url
#     {'text': 'Give me a step by step plan on how I can submit this form. Tell me exactly what to click and type without any other information.'},
# ])
# response, history = model.chat(tokenizer, query=query, history=None)
# print(response)

# # TODO: RCI


# # Map natural language actions to UI actions
# steps = response.strip().split('\n')
# steps = [step for step in steps if step.startswith(tuple("0123456789"))]
# action_types = []
# for step in steps:
#     # TODO: Train an action type classification model or ping another LLM. This one doesn't work.
#     query = tokenizer.from_list_format([
#         {'image': 'tmp/state.png'}, # Either a local path or an url
#         {'text': f"Extract the action type from the following text: '{step}'. Possible values are '<TYPE>' (if the text is similar to typing something into a field) or '<CLICK>' (if the text involves just clicking a button or element). You must return either '<TYPE>' or '<CLICK>', do not explain."},
#     ])
#     response, _ = model.chat(tokenizer, query=query, history=None)
#     print(response)

# action_types = ['<TYPE>', '<TYPE>', '<TYPE>', '<CLICK>']

# # Extract the field names
# # TODO: REGEX or with another model or other LLM.
# field_names = ['Name', 'Email', 'Your answer', 'Submit']

# Find the bounding boxes of the fields
# from transformers import LayoutLMv2Processor
# from PIL import Image

# processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased")

# image = Image.open(
#     "tmp/state.png"
# ).convert("RGB")
# encoding = processor(
#     image, return_tensors="pt"
# ) 
# print(encoding.keys())
# dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'bbox', 'image'])


# TODO: Now that I know the locations of all the text, work on clicking on the right thing.
import easyocr
reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory
result = reader.readtext('tmp/state.png')

from PIL import Image, ImageDraw

# Load the image
image_path = 'tmp/state.png'
image = Image.open(image_path)

# Initialize drawing context
draw = ImageDraw.Draw(image)

# Iterate through all results and draw bounding boxes
for entry in result:
    bbox = entry[0]  # Bounding box coordinates
    text = entry[1]  # Extracted text
    confidence = entry[2]  # Confidence level
    # Draw rectangle
    draw.rectangle([bbox[0][0], bbox[0][1], bbox[2][0], bbox[2][1]], outline="red")
    # Optionally, to include text and confidence on the image, uncomment the following line
    # draw.text((bbox[0][0], bbox[0][1]), f"{text} ({confidence})", fill="red")

# Save the image with bounding boxes
image.save('state_bbox_easyocr.png')



import IPython; IPython.embed()
# for idx, field in enumerate(field_names):
#     # Bounding box drawing
#     query = tokenizer.from_list_format([
#         {'image': 'tmp/state.png'}, # Either a local path or an url
#         {'text': f"框出图中{field}的位置"},
#     ])    
#     response, history = model.chat(tokenizer, query=query, history=None)
#     print(response)
#     image = tokenizer.draw_bbox_on_latest_picture(response, history)
#     if image:
#         image.save(f'data/{idx}.jpg')
#     else:
#         print("no box")
#     import IPython; IPython.embed()
