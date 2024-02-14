from transformers import AutoProcessor, SeamlessM4Tv2Model
import torchaudio

processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")

# from audio
audio, orig_freq =  torchaudio.load("/home/ubuntu/ai/projects/lao_asr_s2tt/test_data/test_lao_short.wav")
audio =  torchaudio.functional.resample(audio, orig_freq=orig_freq, new_freq=16_000) # must be a 16 kHz waveform array
audio_inputs = processor(audios=audio, return_tensors="pt")
audio_array_from_audio = model.generate(**audio_inputs, tgt_lang="eng")[0].cpu().numpy().squeeze()

# TODO: Verify good performance on seamlessv2large model... Odd that I was getting different results in the streaming setting, wondering what's going on. SM4t medium seems to work