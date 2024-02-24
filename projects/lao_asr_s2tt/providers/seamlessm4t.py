from transformers import AutoProcessor, SeamlessM4TForSpeechToText
import torchaudio
from typing import Any, Dict, List
import torch
import sys
sys.path.append("..")
import utils
import constants


class SeamlessM4T:
    def __init__(self, device: str = "cuda", target_lang: str = "lao", chunk_audio=False):
        """
        Initialize the SeamlessM4T class.

        Args:
            device (str, optional): The device to use. Defaults to "cuda".
            target_lang (str, optional): The target language. Defaults to "lao".
                More found here: https://github.com/facebookresearch/seamless_communication/blob/main/demo/m4tv2/lang_list.py
            chunk_audio (bool, optional): Whether to use chunk_audio mode. Defaults to False.
        """
        model_name = "facebook/hf-seamless-m4t-large"
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.target_lang = target_lang
        self.chunk_audio = chunk_audio
        self.model = SeamlessM4TForSpeechToText.from_pretrained(model_name).to(self.device)
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

        if self.chunk_audio:
            preprocessed_audio_arrays = [audio_array.unsqueeze(0) for audio_array in preprocessed_audio_arrays]
            audio_chunks = [utils.chunk_audio(waveform=audio_array, sample_rate=16_000, chunk_size_ms=constants.CHUNK_SIZE_MS, overlap_ms=0) for audio_array in preprocessed_audio_arrays]
            for chunk in audio_chunks:
                out = self.generate_and_decode(chunk)
                out_text = " ".join(out)
                outputs.append({
                    "text": out_text
                })
        else:
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
