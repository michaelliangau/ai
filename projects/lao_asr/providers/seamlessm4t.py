from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText
import torchaudio
from typing import Any, Dict
import torch
class SeamlessM4T:
    def __init__(self):
        """
        Initialize the SeamlessM4T class.
        """
        model_name = "facebook/seamless-m4t-v2-large"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = SeamlessM4Tv2ForSpeechToText.from_pretrained(model_name).to(self.device)

    def forward(self, batch: Dict[str, Any], tgt_lang: str = "lao"):
        """
        Forward pass through the model.

        Args:
            batch (Dict[str, Any]): A dictionary containing the input data.
            tgt_lang (str, optional): The target language. Defaults to "lao".

        Returns:
            numpy.ndarray: The output audio array.
        """
        outputs = []
        audio_arrays = [torch.from_numpy(audio["array"]) for audio in batch["audio"]]
        audio_sampling_rates = [audio["sampling_rate"] for audio in batch["audio"]]
        audio_arrays = [torchaudio.functional.resample(audio_array, orig_freq=sampling_rate, new_freq=16_000) if sampling_rate != 16_000 else audio_array for audio_array, sampling_rate in zip(audio_arrays, audio_sampling_rates)] # SeamlessM4T only supports 16kHz audio
        inputs = self.processor(audios=audio_arrays, return_tensors="pt", sampling_rate=16_000)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        output_tokens = self.model.generate(**inputs, tgt_lang=tgt_lang)
        
        for output_token in output_tokens:
            source = self.processor.decode(output_token, skip_special_tokens=True)
            outputs.append({
                "text": source
            })

        return outputs
