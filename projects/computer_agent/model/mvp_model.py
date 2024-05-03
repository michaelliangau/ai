import torch
import torch.nn as nn
from transformers import BertModel, YolosModel, PreTrainedModel
from torchvision.models import resnet50

class ImageTextModel(PreTrainedModel):
    def __init__(self, config):
        super(ImageTextModel, self).__init__(config)
        self.resnet50 = resnet50(pretrained=True)
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-1])
        self.bert = BertModel.from_pretrained("google-bert/bert-base-uncased")
        self.sigmoid = nn.Sigmoid()
        self.activation = nn.ReLU()
        self.loss = nn.MSELoss()
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )
        self.text_upsampling_layer = nn.Linear(768, 2048)

    def forward(self, image, text, attention_mask, label=None):
        # Image Encoder
        image_features = self.resnet50(image)
        image_features = torch.flatten(image_features, 1)

        # Text Encoder
        text_outputs = self.bert(input_ids=text, attention_mask=attention_mask)
        text_features = self.activation(self.text_upsampling_layer(text_outputs.pooler_output))

        # Combine feature vectors
        combined_features = text_features + image_features

        x = self.fc(combined_features)
        logits = self.sigmoid(x)
        outputs = {'logits': logits}
        if label is not None:
            loss = self.loss(logits, label)
            outputs['loss'] = loss
        return outputs
