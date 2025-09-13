"""Manages the ASR model."""

import io
import torch
import torchaudio
import librosa
import numpy as np
import soundfile as sf
import noisereduce as nr
from transformers import WhisperProcessor, WhisperForConditionalGeneration
# from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

from word_correction import postprocess_text

# test
from faster_whisper import WhisperModel

class ASRManager:

    def __init__(self):
        # Load processor and model
        # Whisper
        self.processor = WhisperProcessor.from_pretrained("./whisper_processor")
        self.model = WhisperForConditionalGeneration.from_pretrained("./tiny-new")
        self.model.generation_config.input_ids = self.model.generation_config.forced_decoder_ids
        self.model.generation_config.forced_decoder_ids = None
        
        # W2V2
        # self.processor = Wav2Vec2Processor.from_pretrained(".")
        # self.model = Wav2Vec2ForCTC.from_pretrained(".")
        # self.model.eval()
        
        if torch.cuda.is_available():
            self.model.to("cuda")
        
        # F Whisper
        # self.model = WhisperModel(".", compute_type="float16", device = "cuda") 
    
    def asr(self, audio_bytes: bytes) -> str:
        # Decode WAV from bytes
        with io.BytesIO(audio_bytes) as audio_buffer:
            audio, sr = sf.read(audio_buffer)
        
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)

        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000

        # Reduce noise
#         audio = nr.reduce_noise(y=audio, sr=sr, prop_decrease=0.65, stationary=False) # Both tensor n numpy works as input, output as numpy

        # Normalise audio to [-1, 1]
        # audio = audio / np.max(np.abs(audio))
        
        # Whisper expects numpy float32
        input_features = self.processor(audio, sampling_rate=16000, return_tensors="pt").input_features

        # # Wav2Vec2
        # input_features = self.processor(audio, sampling_rate=16000, return_tensors="pt", padding="longest").input_values

        if torch.cuda.is_available():
            input_features = input_features.to("cuda")

        # Generate transcription with Whisper
        predicted_ids = self.model.generate(input_features)
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        return transcription

    
    
    
#         audio = audio.astype(np.float32)
#         segments, _ = self.model.transcribe(audio, beam_size=5)
#         output = " ".join(segment.text for segment in segments)
#         print(output)
#         return output
        
        
        
        
        
        # Generate transcription with W2V2
        # Forward pass through the model (no need to use .generate)
        # with torch.no_grad():
        #     logits = self.model(input_features).logits
         
        # Decode the logits to get the predicted ids
#         predicted_ids = torch.argmax(logits, dim=-1)

#         # Decode the predicted ids to text
#         transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
#         return transcription

