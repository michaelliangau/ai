"""
Inference file for SeamlessM4Tv2Model

It's performance on my single test_lao_short.wav file is actually worse than the v1 model.
"""
import torch
import torchaudio
from transformers import AutoProcessor, SeamlessM4Tv2Model

device = torch.device("cuda:0")

processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large").to(device)

# Read an audio file and resample to 16kHz:
audio, orig_freq = torchaudio.load(
    "/home/ubuntu/ai/projects/lao_asr_s2tt/test_data/test_lao_short.wav"
)
audio = torchaudio.functional.resample(
    audio, orig_freq=orig_freq, new_freq=16_000
)  # must be a 16 kHz waveform array
audio_inputs = processor(audios=audio, return_tensors="pt")
audio_inputs = {k: v.to(device) for k, v in audio_inputs.items()}

# from audio
output_tokens = model.generate(**audio_inputs, tgt_lang="eng", generate_speech=False)
translated_text_from_audio = processor.decode(
    output_tokens[0].tolist()[0], skip_special_tokens=True
)

print(translated_text_from_audio)
