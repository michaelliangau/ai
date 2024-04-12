import torch
import torch.nn as nn
from transformers import BertModel, CLIPModel

class CLIPBERTModel(nn.Module):
    def __init__(self):
        super(CLIPBERTModel, self).__init__()
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.bert_model = BertModel.from_pretrained("google-bert/bert-base-uncased")
        self.bert_downsample_layer = nn.Linear(768, 512)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)
    def forward(self, image, text, attention_mask, label):
        # Process Image
        image_embed = self.activation(self.clip_model.get_image_features(image))

        # Process Text
        text_outputs = self.bert_model(input_ids=text, attention_mask=attention_mask)
        text_embed = self.activation(text_outputs.last_hidden_state[:, 0, :])
        text_embed = self.activation(self.bert_downsample_layer(text_embed))

        # Combine features
        combined_features = image_embed + text_embed
        
        # Final linear layer
        x = self.activation(self.fc1(combined_features))
        x = self.activation(self.fc2(x))
        logits = self.sigmoid(self.fc3(x))
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
