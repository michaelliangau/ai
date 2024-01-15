from datasets import load_dataset
from tqdm import tqdm
import jiwer
import matplotlib.pyplot as plt
import os
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--provider", help="Specify the ASR provider to use. Options: 'whisper' or 'seamlessm4t'", choices=['whisper', 'seamlessm4t'], default="whisper")
args = parser.parse_args()

# Hyperparameters
batch_size = 1

if args.provider == "whisper":
    import providers.whisper_v3_large as whisper
    provider = whisper.Whisper(batch_size=batch_size)
elif args.provider == "seamlessm4t":
    import providers.seamlessm4t as seamlessm4t
    provider = seamlessm4t.SeamlessM4T()
else:
    raise ValueError(f"Unknown provider: {args.provider}")


# Download fleurs Lao test
ds = load_dataset("google/fleurs", "lo_la", split="test")


# Run transcriptions
outputs = []
batches = [ds[i:i + batch_size] for i in range(0, len(ds), batch_size)]


for batch in tqdm(batches):
    try:
        transcriptions = batch["transcription"]
        results = provider.forward(batch=batch)
        
        for result, target in zip(results, transcriptions):
            pred = result["text"]
            wer = jiwer.wer(target, pred)
            outputs.append({"prediction": pred, "target": target, "wer": wer})
    except RuntimeError as e:
        print(f"Error: {e}")
        continue

# Generate metrics
wer = sum([output["wer"] for output in outputs]) / len(outputs)

# Generate a WER plot
wer_values = [output["wer"] for output in outputs]
plt.figure(figsize=(10, 5))
plt.hist(wer_values, bins=np.arange(0, 3 + 0.1, 0.1), edgecolor='black')
plt.title(f'{args.provider} WER values')
plt.xlabel('WER')
plt.ylabel('Frequency')
plt.xlim([0, 1.5])
plt.text(0.95, 0.95, f'Mean WER: {wer:.2f}', horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes)

# Save the plot to outputs folder
if not os.path.exists(f'./benchmark_outputs'):
    os.makedirs(f'./benchmark_outputs')
plt.savefig(f'./benchmark_outputs/wer_plot_{args.provider}.png')
