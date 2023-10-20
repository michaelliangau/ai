import datasets
import transformers
import model
import torch
import os
import collator
import tqdm
from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast

import sys
sys.path.append("../..")
import common.utils as common_utils


# Hyperparameters
experiment_name = "dev"
forward_beta = 10.0
forward_num_timesteps = 100
forward_decay_rate = 0.93
num_epochs = 4
batch_size = 12
learning_rate = 4e-3
device = "cuda"
save_steps = 100
do_eval = True
eval_steps = 200


# Outputs folder
common_utils.create_folder("outputs")

# Weights & Biases
common_utils.start_wandb_logging(name=experiment_name, project_name="denoising_diffusion_primitives")

# Device
torch_device = common_utils.get_device(device)

# Tokenizer
tokenizer = transformers.T5TokenizerFast.from_pretrained("t5-small")
text_embedding_model = transformers.T5EncoderModel.from_pretrained("t5-small").to(torch_device)

# Model
unet = model.UNet().to(torch_device)

# Forward/Backward Process
forward_process = model.ForwardProcess(num_timesteps=forward_num_timesteps, initial_beta=forward_beta, decay_rate=forward_decay_rate, torch_device=torch_device)
backward_process = model.BackwardProcess(model=unet, torch_device=torch_device)

# Data
train_ds = datasets.load_dataset('HuggingFaceM4/COCO', '2014_captions')['train']
train_ds = train_ds.remove_columns(['filepath', 'sentids', 'filename', 'imgid', 'split', 'sentences_tokens', 'sentences_sentid', 'cocoid'])
eval_ds = datasets.load_dataset('HuggingFaceM4/COCO', '2014_captions')['validation']
eval_ds = eval_ds.remove_columns(['filepath', 'sentids', 'filename', 'imgid', 'split', 'sentences_tokens', 'sentences_sentid', 'cocoid'])

# Collator
collate_fn = collator.Collator().collate
train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8, drop_last=True)
eval_dataloader = torch.utils.data.DataLoader(eval_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8, drop_last=True)

# Optimizer and Scheduler
optimizer = Adam(list(unet.parameters()), lr=learning_rate)
scheduler_steps = num_epochs * len(train_dataloader)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, total_steps=scheduler_steps, pct_start=0.25)

# GradScaler for mixed precision training
scaler = GradScaler()

# Print the number of trainable parameters in both the unet and the downsample text embedding layer
num_trainable_params_unet = sum(p.numel() for p in unet.parameters() if p.requires_grad)
print(f"Number of trainable parameters in UNet: {num_trainable_params_unet}")

# Train loop
for epoch in tqdm.tqdm(range(num_epochs)):
    print("Epoch:", epoch)
    for i, batch in tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
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
        with autocast():
            predicted_noise = backward_process.predict(image=noised_image, text=mean_text_embedding)

            # Loss
            loss = torch.nn.functional.mse_loss(noise_added, predicted_noise)

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

        # Log to Weights & Biases
        common_utils.log_wandb({
            "loss": loss.item(),
            "learning_rate": scheduler.get_last_lr()[0],
        })

        # Save checkpoint every `save_steps` steps
        if i % save_steps == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, os.path.join("./outputs/", f"checkpoint_{epoch}_{i}.pt"))
        
        # Evaluate every `eval_steps` steps
        if do_eval and i % eval_steps == 0:
            print("Evaluating...")
            unet.eval()
            eval_losses = []
            for j, eval_batch in tqdm.tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
                # Get data
                eval_image = eval_batch["image"].to(torch_device)
                eval_text = eval_batch["sentences_raw"]

                # Forward Noising Step
                eval_timestep = torch.randint(0, forward_num_timesteps, (batch_size,)).to(torch_device)
                eval_noised_image = forward_process.sample(image=eval_image, timestep=eval_timestep)
                eval_noise_added = eval_noised_image - eval_image

                # Backward Generation Step
                eval_inputs = tokenizer(eval_text, return_tensors="pt", padding=True, truncation=True).to(torch_device)
                eval_outputs = text_embedding_model(**eval_inputs)
                eval_text_embedding = eval_outputs.last_hidden_state
                eval_mean_text_embedding = eval_text_embedding.mean(dim=1)
                with autocast():
                    eval_predicted_noise = backward_process.predict(image=eval_noised_image, text=eval_mean_text_embedding)

                # Loss
                eval_loss = torch.nn.functional.mse_loss(eval_noise_added, eval_predicted_noise)
                eval_losses.append(eval_loss.item())

            # Log the mean eval loss over the entire evaluation loop to Weights & Biases
            common_utils.log_wandb({
                "eval_loss": sum(eval_losses) / len(eval_losses),
            })
            unet.train()

    # Save checkpoint every epoch
    torch.save({
        'epoch': epoch,
        'model_state_dict': unet.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, os.path.join("./outputs/", f"checkpoint_{epoch}.pt"))

# End logging
common_utils.end_wandb_logging()