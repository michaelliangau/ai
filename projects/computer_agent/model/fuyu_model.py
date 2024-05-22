from transformers import PreTrainedModel, FuyuForCausalLM, BitsAndBytesConfig, FuyuProcessor, GenerationConfig
import torch
import PIL
import torch.nn as nn


class ImageTextModel(PreTrainedModel):
    def __init__(self, config):
        super(ImageTextModel, self).__init__(config)
        self.processor = FuyuProcessor.from_pretrained("adept/fuyu-8b")
        self.model = FuyuForCausalLM.from_pretrained(
            "adept/fuyu-8b", device_map="cuda:0", torch_dtype=torch.float16
        )
        self.processor = FuyuProcessor.from_pretrained("adept/fuyu-8b")
        self.generation_config = GenerationConfig(output_hidden_states=True, return_dict_in_generate=True)
        self.projection_layer = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2),
        )
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.MSELoss()


    def forward(self, image_patches, input_ids, attention_masks, image_patches_indices, labels=None):
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_masks,
            image_patches=image_patches,
            image_patches_indices=image_patches_indices,
            max_new_tokens=1,
            generation_config=self.generation_config
        )

        # Take the last logic from the last layer. TODO: Key parameter to change is the first -1
        x = outputs["hidden_states"][0][20][:, -1, :]
        logits = self.sigmoid(self.projection_layer(x))
        outputs = {"logits": logits}
        if labels is not None:
            loss = self.loss(logits, labels)
            outputs["loss"] = loss
        return outputs