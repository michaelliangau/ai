from PIL import Image
import requests
import io
from transformers import FuyuForCausalLM, FuyuProcessor
from PIL import ImageDraw
import re

def resize_to_max(image, max_width=1920, max_height=1080):
    width, height = image.size
    if width <= max_width and height <= max_height:
        return image

    scale = min(max_width/width, max_height/height)
    width = int(width*scale)
    height = int(height*scale)

    return image.resize((width, height), Image.LANCZOS)

def pad_to_size(image, canvas_width=1920, canvas_height=1080):
    width, height = image.size
    if width >= canvas_width and height >= canvas_height:
        return image

    # Paste at (0, 0)
    canvas = Image.new("RGB", (canvas_width, canvas_height))
    canvas.paste(image)
    return canvas

def coords_from_response(response):
    # y1, x1, y2, x2
    pattern = r"<box>(\d+),\s*(\d+),\s*(\d+),\s*(\d+)</box>"

    match = re.search(pattern, response)
    if match:
        # Unpack and change order
        y1, x1, y2, x2 = [int(coord) for coord in match.groups()]
        return (x1, y1, x2, y2)
    else:
        print("error")
        
pretrained_path = "adept/fuyu-8b"
processor = FuyuProcessor.from_pretrained(pretrained_path)
model = FuyuForCausalLM.from_pretrained(pretrained_path, device_map='cpu')

bbox_prompt = f"When presented with a box, perform OCR to extract text contained within it. If provided with text, generate the corresponding bounding box.\nEmail"
# bbox_image_url = "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/bbox_sample_image.jpeg"
# bbox_image_pil = Image.open(io.BytesIO(requests.get(bbox_image_url).content))
bbox_image_pil = Image.open("data/hi2.png")
bbox_image_pil = bbox_image_pil.convert("RGB")


padded = resize_to_max(bbox_image_pil)
padded = pad_to_size(padded)

model_inputs = processor(text=bbox_prompt, images=[padded]).to('cpu')

outputs = model.generate(**model_inputs, max_new_tokens=40)
post_processed_bbox_tokens = processor.post_process_box_coordinates(outputs)[0]
decoded = processor.decode(post_processed_bbox_tokens, skip_special_tokens=True)
decoded = decoded.split('\x04', 1)[1] if '\x04' in decoded else ''
coords = coords_from_response(decoded)
print(coords)

# Assuming the coordinates are in the format [x_min, y_min, x_max, y_max]

# Define the box coordinates
box_coordinates = list(coords)


# Create a drawing context on the image
draw = ImageDraw.Draw(bbox_image_pil)

# Draw the bounding box using the coordinates
draw.rectangle(box_coordinates, outline="red", width=2)

# Save the image with the drawn bounding box
bbox_image_pil.save("data/bbox_sample.png")
import IPython; IPython.embed()



# Run through model
# from transformers import FuyuProcessor, FuyuForCausalLM
# from PIL import Image
# import requests

# # load model and processor
# model_id = "adept/fuyu-8b"
# processor = FuyuProcessor.from_pretrained(model_id)
# model = FuyuForCausalLM.from_pretrained(model_id, device_map="cpu")

# # prepare inputs for the model
# text_prompt = "You can interact with this form with the keyboard or mouse. I want you to fill this form with relevant details. You have not taken any actions prior to this yet. What is the first action you should take?\n"
# url = "data/leetcode_og.png"
# url = "data/form.png"

# if url.startswith("http"):
#     image = Image.open(requests.get(url, stream=True).raw)
# else:
#     image = Image.open(url)
#     image = image.convert('RGB')
#     import IPython; IPython.embed()
#     image = image.resize((1000, int(image.height * (1000 / image.width))))

# inputs = processor(text=text_prompt, images=image, return_tensors="pt").to("cpu")

# # autoregressively generate text
# generation_output = model.generate(**inputs, max_new_tokens=50)
# generation_text = processor.batch_decode(generation_output[:, -50:], skip_special_tokens=True)
# print("Generated text:", generation_text)



# Draw bbox
# from PIL import Image, ImageDraw, ImageFont

# # Load the image
# image_path = 'data/form.png'
# image = Image.open(image_path)

# # Create a drawing context
# draw = ImageDraw.Draw(image)

# # Image dimensions
# width, height = image.size

# # Grid dimensions
# grid_width = width // 10
# grid_height = height // 10

# # Line color
# line_color = (255, 0, 0)  # Red

# # Draw vertical lines
# for i in range(1, 10):
#     start_point = (i * grid_width, 0)
#     end_point = (i * grid_width, height)
#     draw.line([start_point, end_point], fill=line_color, width=1)

# # Draw horizontal lines
# for i in range(1, 10):
#     start_point = (0, i * grid_height)
#     end_point = (width, i * grid_height)
#     draw.line([start_point, end_point], fill=line_color, width=1)

# # Optionally, add numbers to the grid
# font_size = 40  # Increased font size
# font = ImageFont.truetype("data/Arial.ttf", font_size)
# for x in range(10):
#     for y in range(10):
#         draw.text((x * grid_width + 5, y * grid_height + 5), f"{x},{y}", fill=line_color, font=font)

# # Save or display the image
# # image.show()  # To display the modified image
# image.save('data/form_with_grid.png')  # To save the modified image