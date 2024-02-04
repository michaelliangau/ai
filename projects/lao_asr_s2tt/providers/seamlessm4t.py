from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText
import torchaudio
from typing import Any, Dict, List
import torch

class SeamlessM4T:
    def __init__(self, device: str = "cuda", target_lang: str = "lao"):
        """
        Initialize the SeamlessM4T class.

        Args:
            target_lang (str, optional): The target language. Defaults to "lao".
                More found here: https://github.com/facebookresearch/seamless_communication/blob/main/demo/m4tv2/lang_list.py
        """
        model_name = "facebook/seamless-m4t-v2-large"
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.target_lang = target_lang
        self.model = SeamlessM4Tv2ForSpeechToText.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def preprocess_audio(self, batch: Dict[str, Any]):
        """
        Preprocess audio data for the model.

        Args:
            batch (Dict[str, Any]): A dictionary containing the input data.

        Returns:
            List[torch.Tensor]: Preprocessed audio arrays.
        """
        audio_arrays = [torch.from_numpy(audio["array"]) for audio in batch["audio"]]
        audio_sampling_rates = [audio["sampling_rate"] for audio in batch["audio"]]
        preprocessed_audio_arrays = [torchaudio.functional.resample(audio_array, orig_freq=sampling_rate, new_freq=16_000) if sampling_rate != 16_000 else audio_array for audio_array, sampling_rate in zip(audio_arrays, audio_sampling_rates)] # SeamlessM4T only supports 16kHz audio
        return preprocessed_audio_arrays

    def forward(self, batch: Dict[str, Any]):
        """
        Forward pass through the model.

        Args:
            batch (Dict[str, Any]): A dictionary containing the input data.

        Returns:
            numpy.ndarray: The output audio array.
        """
        outputs = []
        preprocessed_audio_arrays = self.preprocess_audio(batch)
        inputs = self.processor(audios=preprocessed_audio_arrays, return_tensors="pt", sampling_rate=16_000)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        output_tokens = self.model.generate(**inputs, tgt_lang=self.target_lang)
        
        for output_token in output_tokens:
            source = self.processor.decode(output_token, skip_special_tokens=True)
            outputs.append({
                "text": source
            })

        return outputs

    def generate_and_decode(self, audio_chunks: List[torch.Tensor]) -> List[str]:
        """
        Generate and decode the output from preprocessed audio arrays.

        Args:
            audio_chunks (List[torch.Tensor]): A list of preprocessed audio arrays.

        Returns:
            List[str]: A list of decoded texts.
        """
        inputs = self.processor(audios=audio_chunks, return_tensors="pt", sampling_rate=16_000)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        output_tokens = self.model.generate(**inputs, tgt_lang=self.target_lang)
        decoded_texts = [self.processor.decode(output_token, skip_special_tokens=True) for output_token in output_tokens]
        return decoded_texts
