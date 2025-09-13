"""Manages the ASR model."""

import io
import torch
import torchaudio
import librosa
import numpy as np
import soundfile as sf
import noisereduce as nr
# from transformers import WhisperProcessor, WhisperForConditionalGeneration
# from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

from word_correction import postprocess_text

# test
from faster_whisper import WhisperModel

class ASRManager:

    def __init__(self):
        # Load processor and model        
        # F Whisper
        # self.model = WhisperModel("./", compute_type="float16", device = "cuda", local_files_only = True, flash_attention=True) 
        self.model = WhisperModel("./", compute_type="float16", device = "cuda", local_files_only = True) 

    def asr(self, audio_bytes: bytes) -> str:
        # Decode WAV from bytes
        with io.BytesIO(audio_bytes) as audio_buffer:
            audio, sr = sf.read(audio_buffer)

        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)

        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000  


        audio = audio.astype(np.float32)
        segments, _ = self.model.transcribe(audio, beam_size=5)
        output = " ".join(segment.text for segment in segments)
        return output