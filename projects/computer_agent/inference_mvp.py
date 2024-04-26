import model.mvp_model as mvp_model
import transformers
import utils
import PIL
from PIL import ImageDraw

# Load checkpoint
checkpoint_path = "results/saved_checkpoints/red_dot"
config = transformers.PretrainedConfig()
model = mvp_model.ImageTextModel.from_pretrained(checkpoint_path, config=config)
image_processor = transformers.AutoImageProcessor.from_pretrained("hustvl/yolos-small")
tokenizer = transformers.BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
model.eval()

# Dummy image
utils.create_white_canvas_with_red_dot(
    path="data/tmp/random_red_dot.png"
)

# Run inference over it
image = PIL.Image.open("data/tmp/random_red_dot.png")
image_t = image_processor(images=image, return_tensors="pt")
text = "Click on the red square."
encoded_text = tokenizer(text, return_tensors='pt')
outputs = model(image=image_t["pixel_values"].squeeze().unsqueeze(0), text=encoded_text['input_ids'].squeeze().unsqueeze(0), attention_mask=encoded_text['attention_mask'].squeeze().unsqueeze(0), label=None)
click_x_y = outputs['logits'].squeeze()

# Draw the outputs onto the image and save it
x = int(click_x_y[0].item())
y = int(click_x_y[1].item())
scaled_x = int(click_x_y[0].item() * 1920)
scaled_y = int(click_x_y[1].item() * 1080)
print(f"Click at x: {scaled_x}, y: {scaled_y}")
draw = ImageDraw.Draw(image)
draw.ellipse([(scaled_x - 5, scaled_y - 5), (scaled_x + 5, scaled_y + 5)], fill='blue', outline='blue')
image.save("data/tmp/random_red_dot_predicted.png")

