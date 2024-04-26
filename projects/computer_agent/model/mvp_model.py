import torch
import torch.nn as nn
from transformers import BertModel, YolosModel, PreTrainedModel

class ImageTextModel(PreTrainedModel):
    def __init__(self, config):
        super(ImageTextModel, self).__init__(config)
        self.image_model = YolosModel.from_pretrained("hustvl/yolos-small")
        self.text_model = BertModel.from_pretrained("google-bert/bert-base-uncased")
        self.bert_downsample_layer = nn.Linear(768, 512)
        self.max_pool_2d_layers = nn.Sequential(
            nn.AdaptiveMaxPool2d((2048, 256)),
            nn.AdaptiveMaxPool2d((1024, 128)),
            nn.AdaptiveMaxPool2d((512, 64)),
            nn.AdaptiveMaxPool2d((256, 32)),
            nn.AdaptiveMaxPool2d((128, 16)),
            nn.AdaptiveMaxPool2d((64, 8)),
        )
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc_layers = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, image, text, attention_mask, label):

        # Process Image
        # Use the last hidden state. Pooler output is the output of the CLS token which is used to classify objects.
        image_out = self.image_model(image)
        image_embed = self.activation(image_out.last_hidden_state)

        # Process Text
        text_outputs = self.text_model(input_ids=text, attention_mask=attention_mask)
        text_embed = self.activation(text_outputs.last_hidden_state[:, 0, :])
        text_embed = self.activation(self.bert_downsample_layer(text_embed))

        # Combine features
        pooled_image_embed = self.max_pool_2d_layers(image_embed)
        flattened_image_embed = torch.flatten(pooled_image_embed, start_dim=1)
        combined_features = flattened_image_embed + text_embed
        
        # Final linear layer
        x = self.fc_layers(combined_features)
        logits = self.sigmoid(x)
        outputs = {'logits': logits}
        if label is not None:
            scaled_logits = torch.stack((logits[:,0] * 1920, logits[:,1] * 1080), dim=1)
            loss = self.shortest_distance_to_box(scaled_logits[:, 0], scaled_logits[:, 1], label[:, 0], label[:, 1], label[:, 2], label[:, 3])
            outputs['loss'] = torch.mean(loss)
        return outputs

    def shortest_distance_to_box(self, x, y, xmin, ymin, xmax, ymax):
        """
        Calculate the shortest distance from a point (x, y) to a rectangular box defined by (xmin, ymin, xmax, ymax).

        Args:
            x: x coordinate of the point
            y: y coordinate of the point
            xmin: x coordinate of the top left corner of the box
            ymin: y coordinate of the top left corner of the box
            xmax: x coordinate of the bottom right corner of the box
            ymax: y coordinate of the bottom right corner of the box
        
        Returns:
            The shortest distance from the point to the box.
        """
        dx = torch.max(torch.max(xmin - x, torch.zeros_like(x)), x - xmax)
        dy = torch.max(torch.max(ymin - y, torch.zeros_like(y)), y - ymax)
        return torch.sqrt(dx**2 + dy**2)
