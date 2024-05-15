import ai.projects.computer_agent.model.resnet_bert_model as resnet_bert_model
import transformers
import utils
import PIL
from PIL import ImageDraw
import torchvision.transforms as transforms


PATH_TO_IMAGE = "data/tmp/random_2_dot.png"
# Load checkpoint
checkpoint_path = "results/checkpoint-1410"
config = transformers.PretrainedConfig()
model = resnet_bert_model.ImageTextModel.from_pretrained(checkpoint_path, config=config)
tokenizer = transformers.BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
model.eval()

# Dummy image
# utils.create_white_canvas_with_red_dot(
#     path="data/tmp/random_red_dot.png"
# )
utils.create_white_canvas_with_2_dot(
    path=PATH_TO_IMAGE
)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the input size expected by ResNet50
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for ResNet50
])


# Run inference over it
image = PIL.Image.open(PATH_TO_IMAGE).convert('RGB')
image_t = transform(image)  # Apply the transformation
text = "Click on the red square."
encoded_text = tokenizer(text, return_tensors='pt')
import IPython; IPython.embed()
outputs = model(image=image_t.unsqueeze(0),text=encoded_text["input_ids"], attention_mask=encoded_text['attention_mask'], label=None)
click_x_y = outputs['logits'].squeeze()

# Draw the outputs onto the image and save it
x = int(click_x_y[0].item())
y = int(click_x_y[1].item())
scaled_x = int(click_x_y[0].item() * 1920)
scaled_y = int(click_x_y[1].item() * 1080)
print(f"Click at x: {scaled_x}, y: {scaled_y}")
draw = ImageDraw.Draw(image)
draw.ellipse([(scaled_x - 5, scaled_y - 5), (scaled_x + 5, scaled_y + 5)], fill='blue', outline='blue')
image.save("data/tmp/random_2_dot_predicted.png")

