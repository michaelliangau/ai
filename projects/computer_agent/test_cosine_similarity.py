from transformers import BertModel
import transformers
import torch

model = BertModel.from_pretrained("google-bert/bert-base-uncased")

tokenizer = transformers.BertTokenizer.from_pretrained("google-bert/bert-base-uncased")

text_1 = "Red"
t1 = tokenizer(text_1, return_tensors='pt')['input_ids']
out1 = model(t1).pooler_output

text_2 = "Green"
t2 = tokenizer(text_2, return_tensors='pt')['input_ids']
out2 = model(t2).pooler_output
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

print(cos(out1, out2))
