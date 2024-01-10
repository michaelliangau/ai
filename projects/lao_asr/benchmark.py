import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
from tqdm import tqdm
import jiwer
import matplotlib.pyplot as plt
import os
import numpy as np

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

batch_size = 16
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=batch_size,
    return_timestamps=False,
    torch_dtype=torch_dtype,
    device=device,
)

# result = pipe("./test_data/test_lao.wav")

# Download fleurs Lao test
ds = load_dataset("google/fleurs", "lo_la", split="test")


# Run transcriptions
outputs = []
batches = [ds[i:i + batch_size] for i in range(0, len(ds), batch_size)]

for batch in tqdm(batches):
    audio_arrays = [data["array"] for data in batch["audio"]]
    transcriptions = batch["transcription"]
    results = pipe(audio_arrays, generate_kwargs={"language": "lao"})
    
    for result, target in zip(results, transcriptions):
        pred = result["text"]
        wer = jiwer.wer(target, pred)
        outputs.append({"prediction": pred, "target": target, "wer": wer})

# Generate metrics
wer = sum([output["wer"] for output in outputs]) / len(outputs)
print(f"MEAN WER: {wer}")


# Generate a WER plot
wer_values = [output["wer"] for output in outputs]
plt.figure(figsize=(10, 5))
plt.hist(wer_values, bins=np.arange(min(wer_values), max(wer_values) + 0.1, 0.1), edgecolor='black')
plt.title('Frequency of WER values')
plt.xlabel('WER')
plt.ylabel('Frequency')

# Save the plot to outputs folder
if not os.path.exists('outputs'):
    os.makedirs('outputs')
plt.savefig('outputs/wer_plot.png')