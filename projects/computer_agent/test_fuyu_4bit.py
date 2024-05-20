import requests
from transformers import FuyuForCausalLM, AutoProcessor, BitsAndBytesConfig
from PIL import Image
import torch

# load model, and processor using the sharded version of the model
model_id = "ybelkada/fuyu-8b-sharded"
processor = AutoProcessor.from_pretrained(model_id)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = FuyuForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)

import requests

img_url = 'https://huggingface.co/adept/fuyu-8b/resolve/main/bus.png'
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')


text_prompt = "Generate a coco-style caption.\n"


model_inputs = processor(text=text_prompt, images=raw_image)
for k, v in model_inputs.items():
    if not isinstance(v, list):
        if v.dtype != torch.long:
            v = v.to(torch.float16)
    else:
        v = v[0]
        if v.dtype != torch.long:
            v = v.to(torch.float16)
    model_inputs[k] = v.to("cuda")

generation_output = model.generate(**model_inputs, max_new_tokens=7)
generation_text = processor.batch_decode(generation_output[:, -7:], skip_special_tokens=True)
print(generation_text)

# TODO: Messing with the sharded to see if we can train this