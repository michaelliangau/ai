"""
From my testing with Whisper V3 Large, even whilst specifying the target language as Lao,
the model outputs Thai.
"""

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from typing import Any, Dict, List

class Whisper:
    def __init__(self, device: str = "cuda", batch_size: int = 16, target_lang: str = "lo", model_task: str = "translate"):
        """
        Initialize the Whisper class.

        Language codes are here: https://github.com/openai/whisper/blob/main/whisper/tokenizer.py#L10
        
        Args:
            device (str, optional): The device to use. Defaults to "cuda".
            batch_size (int, optional): The size of the batch. Defaults to 16.
            target_lang (str, optional): The target language. Defaults to "lo".
            model_task (str, optional): The model task. Defaults to "translate". Other
                option is "transcribe".
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
        self.target_lang = target_lang
        self.model_task = model_task

    def forward(self, batch: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Forward pass through the model.

        Args:
            batch (Dict[str, Any]): A batch of data containing audio arrays.

        Returns:
            List[Dict[str, str]]: A list of dictionaries containing the predictions.
        """
        audio_arrays = [data["array"] for data in batch["audio"]]
        output = self.pipe(audio_arrays, generate_kwargs = {"language":f"<|{self.target_lang}|>","task": self.model_task})
        return output
