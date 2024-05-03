import model.mvp_model as mvp_model
import transformers
import utils
import PIL
from PIL import ImageDraw
import torchvision.transforms as transforms

# Load checkpoint
checkpoint_path = "results/saved_checkpoints/resnet50"
config = transformers.PretrainedConfig()
model = mvp_model.ImageTextModel.from_pretrained(checkpoint_path, config=config)
tokenizer = transformers.BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
model.eval()

# Dummy image
utils.create_white_canvas_with_red_dot(
    path="data/tmp/random_red_dot.png"
)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the input size expected by ResNet50
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for ResNet50
])


# Run inference over it
image = PIL.Image.open("data/tmp/random_red_dot.png")
image_t = transform(image)  # Apply the transformation
# text = "Click on the red square."
# encoded_text = tokenizer(text, return_tensors='pt')
outputs = model(image=image_t.unsqueeze(0),text=None, attention_mask=None, label=None)
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

