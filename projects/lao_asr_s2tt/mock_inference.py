"""
Simple inference over async seamlessm4t-v2-large
"""
import providers.seamlessm4t as seamlessm4t
import torch
import utils
import torchaudio

device = torch.device("mps")
model = seamlessm4t.SeamlessM4T(device=device, target_lang="eng")

# Chunk audio
waveform, sample_rate = torchaudio.load("tmp_0.wav")

# Resample to 16kHz
resampled_audio = [torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)]

# Run inference on the chunks
outputs = model.generate_and_decode(
    audio_chunks=resampled_audio
)

# Concatenate the outputs
output_text = " ".join(outputs)
print(output_text)
