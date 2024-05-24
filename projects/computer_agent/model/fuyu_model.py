"""
Even an 80GB A100 card isn't able to finetune this model...
"""

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
        self.projection_layer = nn.Sequential(
            nn.Linear(4096, 8192),
            nn.ReLU(),
            nn.Linear(8192, 8192),
            nn.ReLU(),
            nn.Linear(8192, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2),
        )
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.MSELoss()
        self.set_gradients()

    def set_gradients(self):
        # First, set all parameters to require no gradients
        for param in self.model.parameters():
            param.requires_grad = False

        # # Enable gradients for embedding tokens and vision embed tokens
        # for param in self.model.language_model.model.embed_tokens.parameters():
        #     param.requires_grad = True

        # Enable gradients for layers 0 to 9 in the PersimmonDecoderLayer
        # for i in range(1):
        #     for param in self.model.language_model.model.layers[i].parameters():
        #         param.requires_grad = True

    def forward(self, image_patches, input_ids, attention_masks, image_patches_indices, labels=None):

        outputs = self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_masks,
            image_patches=image_patches,
            image_patches_indices=image_patches_indices,
            position_ids=torch.arange(0, input_ids.shape[1]).unsqueeze(0).to(input_ids.device),
            use_cache=True,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

        x = outputs["hidden_states"][0][18][:, -1, :]
        logits = self.sigmoid(self.projection_layer(x))
        outputs = {"logits": logits}
        if labels is not None:
            loss = self.loss(logits, labels)
            outputs["loss"] = loss
        return outputs