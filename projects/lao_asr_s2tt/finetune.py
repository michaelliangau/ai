"""
Code to finetune SeamlessM4T on a Lao S2TT dataset as a proof of concept.
"""
from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText
import datasets
from tqdm import tqdm
import utils
import torch
import torchaudio

# Hyperparameters
device = "cuda"

# Datasets
lao_ds = datasets.load_dataset("google/fleurs", "lo_la", split="test")
en_ds = datasets.load_dataset("google/fleurs", "en_us", split="test")
en_dict = {item['id']: item['transcription'] for item in en_ds}
lao_ds = lao_ds.map(lambda row: utils.add_translation(row=row, translations=en_dict, key='en_translation'), num_proc=4)

# Load model
model_name = "facebook/seamless-m4t-v2-large"
processor = AutoProcessor.from_pretrained(model_name)
model = SeamlessM4Tv2ForSpeechToText.from_pretrained(model_name).to(device)

audio, sr = torchaudio.load("./test_data/test_lao.wav")
audio_arrays = [audio]
audio_sampling_rates = [sr]
audio_arrays = [torchaudio.functional.resample(audio_array, orig_freq=sampling_rate, new_freq=16_000) if sampling_rate != 16_000 else audio_array for audio_array, sampling_rate in zip(audio_arrays, audio_sampling_rates)] # SeamlessM4T only supports 16kHz audio
inputs = processor(audios=audio_arrays, return_tensors="pt", sampling_rate=16_000)
inputs = {k: v.to(device) for k, v in inputs.items()}
output_tokens = model.generate(**inputs, tgt_lang="eng")

# TODO: Build the rest of the finetuning loop