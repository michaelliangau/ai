import datasets
import transformers
import model
import torch
import os
import collator
import tqdm
from torch.optim import Adam

import sys
sys.path.append("../..")
import common.utils as common_utils

# Outputs folder
common_utils.create_folder("outputs")

# Weights & Biases
common_utils.start_wandb_logging(name='dev', project_name="denoising_diffusion_primitives")

# Device
torch_device = common_utils.get_device()

# Hyperparameters
forward_beta = 10.0
forward_num_timesteps = 100
forward_decay_rate = 0.93
num_epochs = 10
batch_size = 16
learning_rate = 4e-3

# Tokenizer
tokenizer = transformers.T5TokenizerFast.from_pretrained("t5-small")
text_embedding_model = transformers.T5EncoderModel.from_pretrained("t5-small").to(torch_device)

# Model
unet = model.UNet().to(torch_device)

# Forward/Backward Process
forward_process = model.ForwardProcess(num_timesteps=forward_num_timesteps, initial_beta=forward_beta, decay_rate=forward_decay_rate, torch_device=torch_device)
backward_process = model.BackwardProcess(model=unet)

# Data
train_ds = datasets.load_dataset('HuggingFaceM4/COCO', '2014_captions')['train']
train_ds = train_ds.remove_columns(['filepath', 'sentids', 'filename', 'imgid', 'split', 'sentences_tokens', 'sentences_sentid', 'cocoid'])

# Collator
collate_fn = collator.Collator().collate
data_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8)

# Optimizer and Scheduler
optimizer = Adam(list(unet.parameters()), lr=learning_rate)
scheduler_steps = num_epochs * len(data_loader)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=4e-5, max_lr=learning_rate, step_size_up=scheduler_steps//2, step_size_down=scheduler_steps//2, cycle_momentum=False)

# Train loop
for epoch in tqdm.tqdm(range(num_epochs)):
    print("Epoch:", epoch)
    for i, batch in tqdm.tqdm(enumerate(data_loader), total=len(data_loader)):
        # Get data
        image = batch["image"].to(torch_device)
        text = batch["sentences_raw"]

        # Forward Noising Step
        timestep = torch.randint(0, forward_num_timesteps, (batch_size,)).to(torch_device)
        noised_image = forward_process.sample(image=image, timestep=timestep)
        noise_added = noised_image - image

        # Backward Generation Step
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(torch_device)
        outputs = text_embedding_model(**inputs)
        text_embedding = outputs.last_hidden_state
        mean_text_embedding = text_embedding.mean(dim=1)
        predicted_noise = backward_process.predict(image=noised_image, text=mean_text_embedding)

        # Loss
        loss = torch.nn.functional.mse_loss(noise_added, predicted_noise)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        common_utils.log_wandb({
            "loss": loss.item(),
            "learning_rate": scheduler.get_last_lr()[0],
        })

    # Save checkpoint every epoch
    torch.save({
        'epoch': epoch,
        'model_state_dict': unet.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, os.path.join("./outputs/", f"checkpoint_{epoch}.pt"))

# End logging
common_utils.end_wandb_logging()