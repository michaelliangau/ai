import datasets
import transformers
import diffusion
import unet
import torch
import os
import collator
import tqdm
from torch.optim import Adam

import sys

sys.path.append("../..")
import common.utils as common_utils

# Environment variable
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Hyperparameters
experiment_name = "dev"
forward_num_timesteps = 100
num_epochs = 4
batch_size = 1
learning_rate = 4e-3
device = "cuda"
save_steps = 100
do_eval = False
eval_steps = 200

# Outputs folder
common_utils.create_folder("outputs")

# Weights & Biases
common_utils.start_wandb_logging(
    name=experiment_name, project_name="denoising_diffusion_primitives"
)

# Device
torch_device = common_utils.get_device(device)

# Tokenizer
tokenizer = transformers.T5TokenizerFast.from_pretrained("t5-small")
t5_model = transformers.T5EncoderModel.from_pretrained("t5-small").to(torch_device)
t5_model.eval()

# Model TODO: explicitly call all init args
model = unet.UNet(
    attn_heads=1,
    dim_mults=(1, 2, 1),
    memory_efficient=True,
    layer_attns=False,
    device=torch_device,
).to(torch_device)

# Forward Process
gaussian_diffusion = diffusion.GaussianDiffusion(
    num_timesteps=forward_num_timesteps, torch_device=torch_device
)

# Data
train_ds = datasets.load_dataset("HuggingFaceM4/COCO", "2014_captions")["train"]
train_ds = train_ds.remove_columns(
    [
        "filepath",
        "sentids",
        "filename",
        "imgid",
        "split",
        "sentences_tokens",
        "sentences_sentid",
        "cocoid",
    ]
)
eval_ds = datasets.load_dataset("HuggingFaceM4/COCO", "2014_captions")["validation"]
eval_ds = eval_ds.remove_columns(
    [
        "filepath",
        "sentids",
        "filename",
        "imgid",
        "split",
        "sentences_tokens",
        "sentences_sentid",
        "cocoid",
    ]
)

# Collator
collate_fn = collator.Collator().collate
train_dataloader = torch.utils.data.DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=8,
    drop_last=True,
)
eval_dataloader = torch.utils.data.DataLoader(
    eval_ds,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=8,
    drop_last=True,
)

# Optimizer and Scheduler
optimizer = Adam(list(model.parameters()), lr=learning_rate)
scheduler_steps = num_epochs * len(train_dataloader)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=learning_rate, total_steps=scheduler_steps, pct_start=0.25
)

# Print the number of trainable parameters in both the unet and the downsample text embedding layer
num_trainable_params_unet = sum(
    p.numel() for p in model.parameters() if p.requires_grad
)
print(f"Number of trainable parameters in UNet: {num_trainable_params_unet}")

# Train loop
for epoch in tqdm.tqdm(range(num_epochs)):
    print("Epoch:", epoch)
    for i, batch in tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        # Get data
        image = batch["image"].to(torch_device)
        text = batch["sentences_raw"]
        # Forward noising
        timestep = gaussian_diffusion.sample_random_times(
            batch_size=batch_size, device=torch_device
        )
        noised_image, noise = gaussian_diffusion.sample(image=image, timestep=timestep)

        # Encode text
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(
            torch_device
        )
        attention_mask = inputs["attention_mask"].bool()
        with torch.no_grad():
            outputs = t5_model(**inputs)
        encoded_text = outputs.last_hidden_state
        encoded_text = encoded_text.masked_fill(attention_mask.unsqueeze(-1), 0)

        # Backward Denoising - Predict the noise
        output = model(
            image=image,
            encoded_text=encoded_text,
            timestep=timestep,
            text_mask=attention_mask,
        )

        # Loss
        loss = torch.nn.functional.mse_loss(input=output, target=noise)

        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        # Log to Weights & Biases
        common_utils.log_wandb(
            {
                "loss": loss.item(),
                "learning_rate": scheduler.get_last_lr()[0],
            }
        )

        # Save checkpoint every `save_steps` steps
        if i % save_steps == 0 and i != 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                },
                os.path.join("./outputs/", f"checkpoint_{epoch}_{i}.pt"),
            )

        # Evaluate every `eval_steps` steps
        if do_eval and i % eval_steps == 0:
            print("Evaluating...")
            # TODO: Build eval loop

    # Save checkpoint every epoch
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        os.path.join("./outputs/", f"checkpoint_{epoch}.pt"),
    )

# End logging
common_utils.end_wandb_logging()