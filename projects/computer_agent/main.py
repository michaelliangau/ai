# Input: 1920 x 1080 screen with a red dot inside 10 x 10 side, randomly sampled. Text = Tap the red dot.
# Output: Action (click), location x y within the red dot.

import datasets
from transformers import Trainer, TrainingArguments
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
import model.mvp_model as mvp_model

# Load the dataset
ds = datasets.load_from_disk('data/red_dot_dataset')
train_test_split = ds.train_test_split(test_size=0.1)
train_ds = train_test_split['train']
test_ds = train_test_split['test']

# Assuming you have a model, tokenizer, and dataset ready
model = mvp_model.CLIPBERTModel()
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Define a collate function for padding
def collate_fn(batch):

    # TODO: Write collate fn

    # # Tokenize the inputs and labels in the batch
    # inputs = [item[0] for item in batch]
    # labels = [item[1] for item in batch]
    
    # # Padding the inputs to the maximum length in the batch
    # inputs_padded = pad_sequence([torch.tensor(tokenizer.encode(input_text)) for input_text in inputs],
    #                              batch_first=True, padding_value=tokenizer.pad_token_id)
    
    # # Convert labels to a tensor
    # labels_tensor = torch.tensor(labels)
    
    return inputs_padded, labels_tensor

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=8,   # batch size per device during training
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    leanring_rate=5e-5,
)

# Initialize the Trainer with the collate_fn
trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_ds,              # training dataset
    data_collator=collate_fn,            # custom collate function
)

# Start training
trainer.train()
