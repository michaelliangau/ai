"""
Mock streaming behaviour for async seamlessm4t-v2-large
"""
import providers.seamlessm4t as seamlessm4t
import torch
import utils
import torchaudio

device = torch.device("cuda")
model = seamlessm4t.SeamlessM4T(device=device, target_lang="eng")

# Chunk audio
waveform, sample_rate = torchaudio.load("test_data/test_lao.wav")
audio_chunks = utils.chunk_audio(
    waveform=waveform,
    sample_rate=sample_rate,
    chunk_size_ms=2500,
    overlap_ms=0,
)

# Resample to 16kHz
resampled_audio_chunks = [torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(chunk) for chunk in audio_chunks]

# Run inference on the chunks
outputs = model.generate_and_decode(
    audio_chunks=resampled_audio_chunks
)

# Concatenate the outputs
output_text = " ".join(outputs)
print(output_text)
