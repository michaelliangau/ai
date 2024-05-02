import torch
import torch.nn as nn
from transformers import BertModel, YolosModel, PreTrainedModel
from torchvision.models import resnet50

class ImageTextModel(PreTrainedModel):
    def __init__(self, config):
        super(ImageTextModel, self).__init__(config)
        # Initialize ResNet50 as the backbone for image feature extraction
        self.resnet50 = resnet50(pretrained=True)
        # Remove the final fully connected layer
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-1])
        self.sigmoid = nn.Sigmoid()  # Assuming x needs to be defined before calling sigmoid

        # Define additional components for text processing if needed
        # For example, a transformer model for text embedding could be initialized here

    def forward(self, image, text, attention_mask, label=None):
        # Process the image through ResNet50
        image_features = self.resnet50(image)  # this will output the features from the average pooling layer
        x = torch.flatten(image_features, 1)  # Flatten the features into a vector
        
        # Combine or process these image features with your text features
        # Assuming text features are processed separately and you have a method to combine them
        # Example: x = combine_features(image_features, text_features)

        logits = self.sigmoid(x)
        outputs = {'logits': logits}
        if label is not None:
            scaled_logits = torch.stack((logits[:,0] * 1920, logits[:,1] * 1080), dim=1)
            loss = self.shortest_distance_to_box(scaled_logits[:, 0], scaled_logits[:, 1], label[:, 0], label[:, 1], label[:, 2], label[:, 3])
            outputs['loss'] = torch.mean(loss)
        return outputs

    # def shortest_distance_to_box(self, x, y, xmin, ymin, xmax, ymax):
    #     """
    #     Calculate the shortest distance from a point (x, y) to a rectangular box defined by (xmin, ymin, xmax, ymax).

    #     Args:
    #         x: x coordinate of the point
    #         y: y coordinate of the point
    #         xmin: x coordinate of the top left corner of the box
    #         ymin: y coordinate of the top left corner of the box
    #         xmax: x coordinate of the bottom right corner of the box
    #         ymax: y coordinate of the bottom right corner of the box
        
    #     Returns:
    #         The shortest distance from the point to the box.
    #     """
    #     dx = torch.max(torch.max(xmin - x, torch.zeros_like(x)), x - xmax) # TODO: Understand why this doesn't propagate indices
    #     dy = torch.max(torch.max(ymin - y, torch.zeros_like(y)), y - ymax)
    #     return torch.sqrt(dx**2 + dy**2)
