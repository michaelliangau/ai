# Input: 1920 x 1080 screen with a red dot inside 10 x 10 side, randomly sampled. Text = Tap the red dot.
# Output: Action (click), location x y within the red dot.
# Try to finetune FUYU-8B to perform the task.

import datasets
import transformers
import torch
import model.fuyu_model as fuyu_model
import PIL
import os
import uuid
import utils
from transformers import FuyuProcessor

os.environ["WANDB_PROJECT"] = "computer_agent_fuyu"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints

# Load the dataset
ds = datasets.load_from_disk('data/2_dot_dataset')
# Rename dataset columns
ds = ds.rename_column("text", "input_ids")
ds = ds.rename_column("image", "image_patches")
train_test_split = ds.train_test_split(test_size=0.1)
train_ds = train_test_split['train']
test_ds = train_test_split['test']

# Assuming you have a model, tokenizer, and dataset ready
config = transformers.PretrainedConfig()
model = fuyu_model.ImageTextModel(config)
processor = FuyuProcessor.from_pretrained("adept/fuyu-8b")

def collate_fn(batch):
    image_patches, images_patches_indices, input_ids, attention_masks, labels = [], [], [], [], []

    for item in batch:
        image = PIL.Image.open(item['image_patches'])
        text = item['input_ids']
        inputs = processor(text=text, images=image, return_tensors="pt")
        image_patches.append(inputs["image_patches"][0][0])
        images_patches_indices.append(inputs["image_patches_indices"][0])
        input_ids.append(inputs["input_ids"][0])
        attention_masks.append(inputs["attention_mask"][0])
        
        label = [item["label"][0] / 1920, item["label"][1] / 1080]
        label = torch.tensor(label)
        labels.append(label)

    # Stack all lists to create batches
    image_patches = torch.stack(image_patches)
    images_patches_indices = torch.stack(images_patches_indices)
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels)
    attention_masks = torch.stack(attention_masks)

    return {
        "input_ids": input_ids,
        "attention_masks": attention_masks,
        "image_patches": image_patches,
        "image_patches_indices": images_patches_indices,
        "labels": labels,
    }

output_dir = f'./results/{uuid.uuid4()}'
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
print(f"Saving model to {output_dir}")

# Define training arguments
training_args = transformers.TrainingArguments(
    output_dir = output_dir,
    num_train_epochs=10,
    per_device_train_batch_size=8,
    warmup_ratio=0.05,
    weight_decay=0.005,
    logging_dir='./logs',
    logging_steps=10,
    learning_rate=4e-4,
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
