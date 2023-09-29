from transformers import PreTrainedTokenizer
from datasets import Dataset
import numpy as np
import torch
from typing import List, Dict

def preprocess_data(dataset: Dataset, tokenizer: PreTrainedTokenizer, max_seq_length: int) -> Dataset:
    """Preprocesses the dataset for a Next Token Prediction (NTP) task.

    Args:
        dataset (Dataset): The dataset to preprocess.
        tokenizer (PreTrainedTokenizer): The tokenizer to use for tokenization.
        max_seq_length (int): The maximum sequence length for tokenization.

    Returns:
        Dataset: The preprocessed dataset.
    """
    tokenized_texts = tokenizer(dataset['text'], truncation=True, padding='max_length', max_length=max_seq_length)
    dataset['input_ids'] = np.array(tokenized_texts['input_ids'])
    dataset['labels'] = np.array(tokenized_texts['input_ids'][1:] + [-100])  # -100 is often used as a padding value in Hugging Face models
    return dataset


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
