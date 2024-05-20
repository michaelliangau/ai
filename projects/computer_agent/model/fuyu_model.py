from transformers import PreTrainedModel, FuyuForCausalLM, BitsAndBytesConfig, FuyuProcessor
import torch
import PIL


class ImageTextModel(PreTrainedModel):
    def __init__(self, config):
        super(ImageTextModel, self).__init__(config)
        self.processor = FuyuProcessor.from_pretrained("adept/fuyu-8b")
        self.model = FuyuForCausalLM.from_pretrained(
            "adept/fuyu-8b", device_map="cuda:0", torch_dtype=torch.float16
        )
        self.processor = FuyuProcessor.from_pretrained("adept/fuyu-8b")


    def forward(self, batch, label=None):
        text = ["Click the green square."]
        image = PIL.Image.open("/home/michael/ai/projects/computer_agent/data/2_dot/0.png")
        image = [image]

        processed = self.processor(text=text, images=image, return_tensors="pt")
        for key, value in processed.items():
            if key == "image_patches":
                processed[key] = [patch.to("cuda:0").to(torch.float16) for patch in value]
            elif key == "input_ids" or key == "image_patches_indices":
                processed[key] = value.to("cuda:0").long()
            else:
                processed[key] = value.to("cuda:0").to(torch.float16)

        outputs = self.model.generate(**processed, max_new_tokens=7)

        # TODO: Can't seem to get things to work with self.model() so have to use self.model.generate. Let's see if we can train a model using generate?
        # GPU mem requirements are not crazy high with this setting.
