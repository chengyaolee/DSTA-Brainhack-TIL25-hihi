import io
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import base64
import librosa
import soundfile as sf
import numpy as np
import noisereduce as nr

from faster_whisper import WhisperModel

wav_file_path = "/home/jupyter/advanced/asr/sample_0.wav"

with open(wav_file_path, "rb") as wav_file:
        # Read the binary data from the file
    wav_data = wav_file.read()
        
        # Encode the binary data into base64
    base64_encoded = base64.b64encode(wav_data).decode("utf-8")

audio_bytes = base64.b64decode(base64_encoded)

model = WhisperModel("/home/jupyter/til-25-hihi/asr/whisper_ct2", compute_type="float16", device = "cuda") 
    
with io.BytesIO(audio_bytes) as audio_buffer:
    waveform, sample_rate = torchaudio.load(audio_buffer)

# Whisper expects 16kHz
if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)

# Convert to mono if stereo
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

waveform = waveform.numpy().astype(np.float32)
segments, _ = model.transcribe(waveform, beam_size=5)
output = []
for s in segments:
    output += s.text
        
print(transcription)