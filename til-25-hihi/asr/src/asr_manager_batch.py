"""Manages the ASR model."""

import io
import torch
import torchaudio
import librosa
import numpy as np
import soundfile as sf
import noisereduce as nr
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import base64

# test
from faster_whisper import WhisperModel

class ASRManager:

    def __init__(self):
        # F Whisper
        self.model = WhisperModel("./", compute_type="int8_float16", device = "cuda", local_files_only = True) 
    
    def asr(self, audio_batch: list[dict[str, str]]) -> list[str]:
        
        results = []
        for inst in audio_batch:
            audio_bytes = base64.b64decode(inst["b64"])
            with io.BytesIO(audio_bytes) as audio_buffer:
                audio, sr = sf.read(audio_buffer)

            if audio.ndim == 2:
                audio = np.mean(audio, axis=1)

            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000

            audio = audio.astype(np.float32)

            # Call model on single input — still more efficient than restarting each call
            segments, _ = self.model.transcribe(audio, beam_size=5)
            output = " ".join(segment.text for segment in segments)

            results.append(output)
        return results
