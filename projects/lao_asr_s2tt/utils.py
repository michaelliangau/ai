from typing import Dict, Any, Tuple, List
import torchaudio
import math
import torch

def add_translation(row: Dict[str, Any], translations: Dict[str, Any], key: str) -> Dict:
    """
    Add translation to each item in the dataset
    
    Args:
        row (Dict[str, Any]): A dictionary containing the example data
        translations (Dict[str, Any]): A dictionary containing the translations
        key (str): The key to use for the translation
    
    Returns:
        row: A dictionary containing the data with the translation
            added
    """
    row[key] = translations.get(row['id'], None)
    return row

def chunk_audio(file_path: str, chunk_size_ms: int = 500, overlap_ms: int = 0) -> Tuple[List[torch.Tensor], int]:
    """
    Load audio file and split it into chunks of specified millisecond length with
    optional overlapping.

    Args:
        file_path (str): The path to the audio file.
        chunk_size_ms (int): The size of each chunk in milliseconds.
        overlap_ms (int): The size of the overlap between chunks in milliseconds.

    Returns:
        Tuple[List[torch.Tensor], int]: A tuple containing a list of audio chunks and
            the sample rate.
    """
    waveform, sample_rate = torchaudio.load(file_path)
    chunk_size = int(sample_rate * (chunk_size_ms / 1000))
    overlap_size = int(sample_rate * (overlap_ms / 1000))
    step_size = chunk_size - overlap_size

    chunks = []
    for start in range(0, waveform.size(1) - overlap_size, step_size):
        end = min(start + chunk_size, waveform.size(1))
        chunks.append(waveform[:, start:end])

    return chunks, sample_rate