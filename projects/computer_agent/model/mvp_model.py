import torch
import torch.nn as nn
from transformers import BertModel, YolosModel, PreTrainedModel
from torchvision.models import resnet50

class ImageTextModel(PreTrainedModel):
    def __init__(self, config):
        super(ImageTextModel, self).__init__(config)
        self.resnet50 = resnet50(pretrained=True)
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-1])
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.MSELoss()
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, image, text, attention_mask, label=None):
        image_features = self.resnet50(image)
        x = torch.flatten(image_features, 1)
        x = self.fc(x)
        logits = self.sigmoid(x)
        outputs = {'logits': logits}
        if label is not None:
            loss = self.loss(logits, label)
            outputs['loss'] = loss
        return outputs
