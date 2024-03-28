import torchaudio
from transformers import AutoProcessor, SeamlessM4TModel

processor = AutoProcessor.from_pretrained("facebook/hf-seamless-m4t-medium")
model = SeamlessM4TModel.from_pretrained("facebook/hf-seamless-m4t-medium")

# Read an audio file and resample to 16kHz:
audio, orig_freq = torchaudio.load(
    "/home/ubuntu/ai/projects/lao_asr_s2tt/test_data/test_lao_short.wav"
)
audio = torchaudio.functional.resample(
    audio, orig_freq=orig_freq, new_freq=16_000
)  # must be a 16 kHz waveform array
audio_inputs = processor(audios=audio, return_tensors="pt")

# from audio
output_tokens = model.generate(**audio_inputs, tgt_lang="eng", generate_speech=False)
translated_text_from_audio = processor.decode(
    output_tokens[0].tolist()[0], skip_special_tokens=True
)
print(translated_text_from_audio)
