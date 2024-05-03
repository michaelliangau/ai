# Input: 1920 x 1080 screen with a red dot inside 10 x 10 side, randomly sampled. Text = Tap the red dot.
# Output: Action (click), location x y within the red dot.

import datasets
import transformers
import torch
import model.mvp_model as mvp_model
import PIL
import os
import torchvision.transforms as transforms

# Load the dataset
ds = datasets.load_from_disk('data/red_dot_dataset')
train_test_split = ds.train_test_split(test_size=0.1)
train_ds = train_test_split['train']
test_ds = train_test_split['test']

# Assuming you have a model, tokenizer, and dataset ready
config = transformers.PretrainedConfig()
model = mvp_model.ImageTextModel(config)
tokenizer = transformers.BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
# image_processor = transformers.AutoImageProcessor.from_pretrained("hustvl/yolos-small")
# Define the image transformations for ResNet50
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the input size expected by ResNet50
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for ResNet50
])

def collate_fn(batch):
    images, texts, attention_masks, labels = [], [], [], []

    for item in batch:
        image = PIL.Image.open(item['image']).convert('RGB')  # Ensure image is in RGB
        image = transform(image)  # Apply the transformation
        images.append(image)
        
        text = item['text']
        encoded_text = tokenizer(text, return_tensors='pt')
        texts.append(encoded_text['input_ids'].squeeze())
        attention_masks.append(encoded_text['attention_mask'].squeeze())
        label = [item["label"][0] / 1920, item["label"][1] / 1080]
        label = torch.tensor(label)
        labels.append(label)

    # Stack all lists to create batches
    images = torch.stack(images)
    texts = torch.stack(texts)
    attention_masks = torch.stack(attention_masks)
    labels = torch.stack(labels)

    return {"image": images, "text": texts, "attention_mask": attention_masks, "label": labels}


# Define training arguments
training_args = transformers.TrainingArguments(
    output_dir='./results',
    num_train_epochs=100,
    per_device_train_batch_size=32,
    warmup_ratio=0.1,
    weight_decay=0.005,
    logging_dir='./logs',
    logging_steps=10,
    learning_rate=4e-3,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    save_total_limit=10,
    load_best_model_at_end=True,
    dataloader_num_workers=os.cpu_count(),  # Set the number of workers to the number of CPUs
    gradient_accumulation_steps=1,
    fp16=True,
)

# Initialize the Trainer with the collate_fn
trainer = transformers.Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    data_collator=collate_fn,
)

# Start training
trainer.train()
