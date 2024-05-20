from transformers import PreTrainedModel, FuyuForCausalLM, BitsAndBytesConfig
import torch


class ImageTextModel(PreTrainedModel):
    def __init__(self, config):
        super(ImageTextModel, self).__init__(config)
        # quantization_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_compute_dtype=torch.float16
        # )
        # self.model = FuyuForCausalLM.from_pretrained("ybelkada/fuyu-8b-sharded", quantization_config=quantization_config)

    def forward(self, image, image_patches_indices, text, attention_mask, label=None):
        import IPython; IPython.embed()
        outputs = self.model.generate(
            input_ids=text[0],
            image_patches=image[0],
            image_patches_indices=image_patches_indices[0],
            attention_mask=attention_mask[0],
            # output_hidden_states=True,
            max_new_tokens=7
        )

        pass
