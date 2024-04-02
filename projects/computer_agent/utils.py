from PIL import Image
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