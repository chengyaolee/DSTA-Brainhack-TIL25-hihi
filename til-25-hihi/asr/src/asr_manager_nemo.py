"""Manages the ASR model."""

import io
import torch
import torchaudio
import librosa
import numpy as np
import soundfile as sf
from nemo.collections.asr.models import EncDecCTCModel


class ASRManager:

    def __init__(self):        
        # F Whisper
        self.model = EncDecCTCModel.restore_from("./Speech_To_Text_Finetuning.nemo") 
    
    def asr(self, audio_bytes: bytes) -> str:
        # Decode WAV from bytes
        with io.BytesIO(audio_bytes) as audio_buffer:
            audio, sr = sf.read(audio_buffer)
        
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)

        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000

        transcription = self.model.transcribe([audio])[0]
        return transcription