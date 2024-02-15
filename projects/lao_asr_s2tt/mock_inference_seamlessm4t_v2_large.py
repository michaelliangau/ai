import torch
from transformers import AutoProcessor, SeamlessM4Tv2Model
import torchaudio
device = torch.device("cuda:0")

processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large").to(device)
# from audio
audio, orig_freq =  torchaudio.load("/home/ubuntu/ai/projects/lao_asr_s2tt/test_data/test_lao_short.wav")
import IPython; IPython.embed()
audio =  [torchaudio.functional.resample(audio, orig_freq=orig_freq, new_freq=16_000)] # must be a 16 kHz waveform array
inputs = processor(audios=audio, return_tensors="pt", sampling_rate=16_000)
inputs = {k: v.to(device) for k, v in inputs.items()}
output_tokens = model.generate(**inputs, tgt_lang="eng")

# TODO: Inference isn't working, need to fix this...weird.
output_tokens = output_tokens[0][0]
decoded_texts = [processor.decode(output_token, skip_special_tokens=True) for output_token in output_tokens]

print(decoded_texts)