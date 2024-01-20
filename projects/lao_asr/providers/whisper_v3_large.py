"""
From my testing with Whisper V3 Large, even whilst specifying the target language as Lao,
the model outputs Thai.
"""

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from typing import Any, Dict, List

class Whisper:
    def __init__(self, device: str = "cuda", batch_size: int = 16):
        """
        Initialize the Whisper class.

        Args:
            device (str, optional): The device to use. Defaults to "cuda".
            batch_size (int, optional): The size of the batch. Defaults to 16.
        """
        self.device = device
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model_id = "openai/whisper-large-v3"
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=batch_size,
            return_timestamps=False,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

    def forward(self, batch: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Forward pass through the model.

        Args:
            batch (Dict[str, Any]): A batch of data containing audio arrays.

        Returns:
            List[Dict[str, str]]: A list of dictionaries containing the predictions.
        """
        audio_arrays = [data["array"] for data in batch["audio"]]
        return self.pipe(audio_arrays)
