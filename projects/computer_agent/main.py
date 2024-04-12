# Input: 1920 x 1080 screen with a red dot inside 10 x 10 side, randomly sampled. Text = Tap the red dot.
# Output: Action (click), location x y within the red dot.

import datasets
import transformers
import torch
import model.mvp_model as mvp_model
import PIL

# Load the dataset
ds = datasets.load_from_disk('data/red_dot_dataset')
train_test_split = ds.train_test_split(test_size=0.1)
train_ds = train_test_split['train']
test_ds = train_test_split['test']

# Assuming you have a model, tokenizer, and dataset ready
model = mvp_model.CLIPBERTModel()
tokenizer = transformers.BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
image_processor = transformers.CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def collate_fn(batch):
    images, texts, attention_masks, labels = [], [], [], []

    for item in batch:
        image = PIL.Image.open(item['image'])
        image_t = image_processor(images=image, return_tensors="pt")
        images.append(image_t.pixel_values.squeeze())
        text = item['text']
        encoded_text = tokenizer(text, return_tensors='pt')
        texts.append(encoded_text['input_ids'].squeeze())
        attention_masks.append(encoded_text['attention_mask'].squeeze())
        labels.append(item['label'])
    images = torch.stack(images)
    texts = torch.stack(texts)
    attention_masks = torch.stack(attention_masks)
    labels = torch.tensor(labels)
    return {"image": images, "text": texts, "attention_mask": attention_masks, "label": labels}

# Define training arguments
training_args = transformers.TrainingArguments(
    output_dir='./results',
    num_train_epochs=100,
    per_device_train_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    learning_rate=4e-3,
    save_strategy="epoch",
    save_total_limit=10,
    load_best_model_at_end=True,
)

# Initialize the Trainer with the collate_fn
trainer = transformers.Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_ds,              # training dataset
    data_collator=collate_fn,            # custom collate function
)

# Start training
trainer.train()