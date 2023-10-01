from transformers import PreTrainedTokenizer
from datasets import Dataset
import pandas as pd
import torch
import IPython
from typing import List, Dict

def preprocess_data(dataset: Dataset, tokenizer: PreTrainedTokenizer, max_seq_length: int) -> Dataset:
    """Preprocesses the dataset for a Next Token Prediction (NTP) task.

    This is a fan out operation, more output rows are created than input rows. Having
    batch mode and remove_columns in the map function is critical to this succeeding.
    
    Args:
        dataset (Dataset): The dataset to preprocess.
        tokenizer (PreTrainedTokenizer): The tokenizer to use for tokenization.
        max_seq_length (int): The maximum sequence length for tokenization.

    Returns:
        Dataset: The preprocessed dataset.
    """
    tokenized_texts = tokenizer(dataset['text'], truncation=True, padding='max_length', max_length=max_seq_length)
    new_dataset = []
    tokens = tokenized_texts['input_ids'][0]
    for i, _ in enumerate(tokens):
        if i != 0:
            input_ids = tokens[:i]
            labels = [tokens[i]]
            new_dataset.append({'input_ids': input_ids, 'labels': labels})
    result = {
        'input_ids': [torch.tensor(item['input_ids']) for item in new_dataset],
        'labels': [torch.tensor(item['labels']) for item in new_dataset]
    }
    return result


def collate_fn(batch: List[dict]) -> Dict[str, torch.Tensor]:
    """Collates the input batch into a dictionary of tensors.

    Args:
        batch (list): The input batch to collate.

    Returns:
        dict: A dictionary containing the input_values and labels tensors.
    """
    input_values = torch.tensor([item['input_values'] for item in batch])
    labels = torch.tensor([item['labels'] for item in batch])
    return {'input_values': input_values, 'labels': labels}
