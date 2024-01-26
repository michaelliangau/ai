import datasets
from tqdm import tqdm
import jiwer
import matplotlib.pyplot as plt
import os
import numpy as np
import json
import torch
import argparse
import nltk

# Save the plot to outputs folder
if not os.path.exists(f'./benchmark_outputs'):
    os.makedirs(f'./benchmark_outputs')

parser = argparse.ArgumentParser()
parser.add_argument("--provider", help="Specify the ASR provider to use. Options: 'whisper' or 'seamlessm4t'", choices=['whisper-s2t-lao', 'whisper-s2tt-eng', 'seamlessm4t-s2t-lao', 'seamlessm4t-s2tt-eng'], default="whisper-s2tt-eng")
parser.add_argument("--device", help="Specify the device to use. Options: 'cpu' or 'cuda'", choices=['cpu', 'cuda'], default="cuda")
parser.add_argument("--model_task", help="Specify the model task to use. Options: 'asr' or 's2tt'", choices=['asr', 's2tt'], default="s2tt")
args = parser.parse_args()

# Hyperparameters
batch_size = 12
device = torch.device(args.device)

if args.provider == "whisper-s2t-lao":
    import providers.whisper_v3_large as whisper
    provider = whisper.Whisper(device=device, batch_size=batch_size, model_task="transcribe", target_lang="lo")
elif args.provider == "whisper-s2tt-eng":
    import providers.whisper_v3_large as whisper
    provider = whisper.Whisper(device=device, batch_size=batch_size, model_task="translate", target_lang="lo")
elif args.provider == "seamlessm4t-s2t-lao":
    import providers.seamlessm4t as seamlessm4t
    provider = seamlessm4t.SeamlessM4T(device=device, target_lang="lao")
elif args.provider == "seamlessm4t-s2tt-eng":
    import providers.seamlessm4t as seamlessm4t
    provider = seamlessm4t.SeamlessM4T(device=device, target_lang="eng")
else:
    raise ValueError(f"Unknown provider: {args.provider}")

# Datasets
lao_ds = datasets.load_dataset("google/fleurs", "lo_la", split="test")
en_ds = datasets.load_dataset("google/fleurs", "en_us", split="test")

# Find the matching english transcription with the same id and add it to the lao_ds column
en_dict = {item['id']: item['transcription'] for item in en_ds}

def add_en_translation(example):
    """
    Add English translation to each item in the Lao dataset
    
    Args:
        example: A dictionary containing the example data
    
    Returns:
        example: A dictionary containing the example data with the English translation
            added
    """
    example['en_translation'] = en_dict.get(example['id'], None)
    return example

# Apply the function to the Lao dataset
lao_ds = lao_ds.map(add_en_translation, num_proc=4)

# Run transcriptions
outputs = []
batches = [lao_ds[i:i + batch_size] for i in range(0, len(lao_ds), batch_size)]

for batch in tqdm(batches):
    try:
        transcriptions = batch["transcription"]
        en_translations = batch["en_translation"]
        results = provider.forward(batch=batch)

        for result, target, en_translation in zip(results, transcriptions, en_translations):
            pred = result["text"]
            if args.model_task == "asr":
                cer = jiwer.cer(target, pred)
                bleu = None
            elif args.model_task == "s2tt":
                cer = None                
                pred_norm = ''.join(e for e in pred if e.isalnum() or e.isspace()).lower().strip()
                en_translation_norm = ''.join(e for e in en_translation if e.isalnum() or e.isspace()).lower().strip()
                pred_words = pred_norm.split()
                en_translation_words = en_translation_norm.split()
                bleu = nltk.translate.bleu_score.sentence_bleu(en_translation_words, pred_words, weights=(0.33, 0.33, 0.33), smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method1)

            outputs.append({"prediction": pred, "target": target, "cer": cer, "bleu": bleu})
        break
    except RuntimeError as e:
        print(f"Error: {e}")
        continue

# Save raw outputs
with open(f'./benchmark_outputs/raw_outputs_{args.provider}.json', 'w') as f:
    json.dump(outputs, f, ensure_ascii=False)

# Generate metrics
cer_total = sum([output["cer"] for output in outputs if output["cer"] is not None])
cer_count = len([output for output in outputs if output["cer"] is not None])
cer = cer_total / cer_count if cer_count > 0 else 0
cer_values = [output["cer"] for output in outputs if output["cer"] is not None]
if cer_values:
    plt.figure(figsize=(10, 5))
    plt.hist(cer_values, bins=np.arange(0, max(cer_values) + 0.1, 0.1), edgecolor='black')
    plt.title(f'{args.provider} CER values')
    plt.xlabel('CER')
    plt.ylabel('Frequency')
    plt.xlim([0, max(cer_values)])
    plt.text(0.95, 0.95, f'Mean CER: {cer:.2f}', horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes)
    plt.savefig(f'./benchmark_outputs/cer_plot_{args.provider}.png')

bleu_total = sum([output["bleu"] for output in outputs if output["bleu"] is not None])
bleu_count = len([output for output in outputs if output["bleu"] is not None])
bleu = bleu_total / bleu_count if bleu_count > 0 else 0
bleu_values = [output["bleu"] for output in outputs if output["bleu"] is not None]
if bleu_values:
    plt.figure(figsize=(10, 5))
    plt.hist(bleu_values, bins=np.arange(0, max(bleu_values) + 0.001, 0.001), edgecolor='black')  # Decreased bin size to 0.001
    plt.title(f'{args.provider} BLEU score values')
    plt.xlabel('BLEU score')
    plt.ylabel('Frequency')
    plt.xlim([0, max(bleu_values)])
    plt.text(0.95, 0.95, f'Mean BLEU score: {bleu:.4f}', horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes)  # Increased decimal places to 4
    plt.savefig(f'./benchmark_outputs/bleu_plot_{args.provider}.png')
