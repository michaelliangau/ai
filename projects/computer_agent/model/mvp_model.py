import torch
import torch.nn as nn
from transformers import BertModel, YolosModel, PreTrainedModel
from torchvision.models import resnet50

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate):
        super(TransformerBlock, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout_rate)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, query, key, value):
        attn_output, _ = self.mha(query, key, value)
        x = query + self.dropout(attn_output)  # Apply residual connection and dropout
        x = self.norm(x)  # Apply normalization
        return x
    
class ImageTextModel(PreTrainedModel):
    def __init__(self, config):
        super(ImageTextModel, self).__init__(config)
        self.resnet50 = resnet50(pretrained=True)
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-1])
        self.bert = BertModel.from_pretrained("google-bert/bert-base-uncased")
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.MSELoss()
        self.fc = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2)
        )
        self.fc_text = nn.Sequential(
            nn.Linear(768, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
        )
        self.fc_image = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU()
        )
        self.cross_attn_block = TransformerBlock(embed_dim=1, num_heads=1, dropout_rate=0.1)

    def forward(self, image, text, attention_mask, label=None):
        # Image Encoder
        image_features = self.resnet50(image)
        image_features = torch.flatten(image_features, 1)
        image_features = self.fc_image(image_features)

        # Text Encoder
        text_outputs = self.bert(input_ids=text, attention_mask=attention_mask)
        text_features = self.fc_text(text_outputs.pooler_output)

        # Combine feature vectors
        # TODO: Figuring out how to combine the features effectively...
        combined_features = self.cross_attn_block(
            query=text_features.unsqueeze(-1),
            key=image_features.unsqueeze(-1),
            value=image_features.unsqueeze(-1),
        )
        combined_features = combined_features.squeeze(-1)

        x = self.fc(combined_features)
        logits = self.sigmoid(x)
        outputs = {'logits': logits}
        if label is not None:
            loss = self.loss(logits, label)
            outputs['loss'] = loss
        return outputs
