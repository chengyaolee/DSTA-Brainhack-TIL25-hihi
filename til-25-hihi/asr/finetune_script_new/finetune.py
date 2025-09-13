from datasets import load_dataset
from datasets import Audio
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer

import torchaudio.transforms as T

import string
import re
import librosa
import numpy as np
import evaluate
import random
import torch
import torchaudio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

dataset = load_dataset("json", data_files="./asr_edited.jsonl", split="train")
# dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# def stereo_to_mono(example):
#     audio_array = example["audio"]["array"]
#     # If stereo (2D), convert to mono
#     if audio_array.ndim > 1:
#         audio_array = np.mean(audio_array, axis=0)
#         example["audio"]["array"] = audio_array
#     return example

# dataset = dataset.map(stereo_to_mono)

dataset = dataset.cast_column("audio", Audio(decode=True))
def preprocess_audio(example):
    audio = example["audio"]
    array = audio["array"]
    sr = audio["sampling_rate"]

    # Convert to mono
    if array.ndim > 1:
        array = np.mean(array, axis=0)

    # Resample to 16kHz if needed
    if sr != 16000:
        array = torchaudio.functional.resample(torch.tensor(array), orig_freq=sr, new_freq=16000).numpy()
        sr = 16000

    example["audio"]["array"] = array
    example["audio"]["sampling_rate"] = sr
    return example

dataset = dataset.map(preprocess_audio)

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small.en")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small.en")
processor = WhisperProcessor.from_pretrained("openai/whisper-small.en")

## Resample
def prepare_train_dataset(batch):
    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["transcript"]).input_ids
    return batch

def prepare_test_dataset(batch):
    # # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["transcript"]).input_ids
    return batch

split_dataset = dataset.train_test_split(test_size=0.2, seed=42)

# Access splits
train_dataset = split_dataset["train"]
test_dataset = split_dataset["test"]


train_dataset = train_dataset.map(prepare_train_dataset)
test_dataset = test_dataset.map(prepare_test_dataset,remove_columns=test_dataset.column_names)

model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-small.en",
    attention_dropout=0.1,
    mask_time_prob=0.3,  # 0.05
    mask_time_length=10, # 10
    mask_feature_prob=0.2, # 0
    mask_feature_length=10, # 10
)
# model.generation_config.language = "english"
# model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None
# model.generation_config.use_cache = False
# model.generation_config.generation_max_length=225

# Augmentation Probabilities
REVERB_PROB = 0.25     # Probability to apply reverberation
REVERB_PROFILE = [
    ["reverb", "-w", "50", "75", "100"], # Wet-only reverb with varying room scales
    ["reverb", "50", "50", "75"],       # Different reverb params
]

PITCH_SHIFT_PROB = 0.25
PITCH_SHIFT_SEMITONES_MIN = -1.5 # Min semitones to shift
PITCH_SHIFT_SEMITONES_MAX = 1.5  # Max semitones to shift

TEMPO_PERTURB_PROB = 0.3 # Probability to apply tempo perturbation
TEMPO_RATE_MIN = 0.9          # Min rate for time stretching
TEMPO_RATE_MAX = 1.1         # Max rate for time stretching

NOISE_PROB = 0.5 # Probability for Gaussian noise
NOISE_LEVEL = 0.003    # Level of Gaussian noise


@dataclass
class DataCollatorSpeechSeq2SeqWithAugment:
    processor: WhisperProcessor
    decoder_start_token_id: int
    noise_level: float = 0.005 
    
    def add_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(waveform) * self.noise_level
        return waveform + noise

    def time_stretch(self, audio, rate=1):
        return librosa.effects.time_stretch(audio, rate=rate)

    def pitch_shift(self, audio, sr, n_steps_range=(-2, 2)):
        n_steps = np.random.uniform(*n_steps_range)
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

    def augment_audio(self, audio, sr=16000):
        if random.random() < 0.5:
            audio = self.add_noise(audio)
        audio_np = audio.detach().cpu().numpy()
        if random.random() < 0.3:
            random_rate = random.uniform(0.9, 1.1)
            audio_np = self.time_stretch(audio_np,random_rate)
        if random.random() < 0.3:
            audio_np = self.pitch_shift(audio_np, sr)
        audio = torch.from_numpy(audio_np).float().to(audio.device)
        return audio

    def spec_augment(self, mel_spec, time_mask_param=30, freq_mask_param=13, prob=0.5):
        if random.random() > prob:
            return mel_spec  # no masking

        # Apply 2 masks each
        for _ in range(2):
            mel_spec = T.TimeMasking(time_mask_param=time_mask_param)(mel_spec)
            mel_spec = T.FrequencyMasking(freq_mask_param=freq_mask_param)(mel_spec)

        return mel_spec

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # print("Batch keys:", features[0].keys())
        if "input_features" not in features[0]:
            input_features = []
            for feature in features:
                waveform = torch.tensor(feature["audio"]["array"])
                sampling_rate = feature["audio"]["sampling_rate"]

                # Add noise
                noisy_waveform = self.augment_audio(waveform)

                # Re-extract features from noisy waveform
                inputs = self.processor.feature_extractor(
                    noisy_waveform.numpy(), sampling_rate=sampling_rate, return_tensors="pt"
                )
                augmented_inputs = self.spec_augment(inputs.input_features[0])
                input_features.append({"input_features": augmented_inputs})

        else:
            # Evaluation mode — use precomputed features
            input_features = [{"input_features": f["input_features"]} for f in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return {
            "input_features": batch["input_features"],
            "labels": batch["labels"],
        }
        
@dataclass
class DataCollatorSpeechSeq2SeqWithPaddingAndNoise:
    processor: WhisperProcessor
    decoder_start_token_id: int
    
    # Reverb
    reverb_prob: float = 0.0
    reverb_profiles: List[List[str]] = field(default_factory=list)
    
    # Pitch shifting
    pitch_shift_prob: float = 0.0
    pitch_shift_semitones_min: float = 0.0
    pitch_shift_semitones_max: float = 0.0
    
    # Tempo perturbation
    tempo_perturb_prob: float = 0.0
    tempo_rate_min: float = 1.0
    tempo_rate_max: float = 1.0
    
    # Gaussian noise
    noise_prob: float = 0.0
    noise_level: float = 0.005
    
        
    def apply_reverb(self, waveform_tensor: torch.Tensor, sampling_rate: int) -> torch.Tensor:
        if (not self.reverb_profiles) or (random.random() > self.reverb_prob):
            return waveform_tensor
        
        profile = random.choice(self.reverb_profiles)
        try:
            # torchaudio.sox_effects expects a 2D tensor [channels, samples]
            if waveform_tensor.ndim == 1:
                waveform_tensor = waveform_tensor.unsqueeze(0)
            
            reverbed_waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
                waveform_tensor, sampling_rate, [profile], channels_first=True
            )
            return reverbed_waveform.squeeze(0) # Back to 1D if it was originally
        except Exception as e:
            print(f"Warning: Could not apply reverb: {e}")
            return waveform_tensor.squeeze(0) if waveform_tensor.ndim > 1 else waveform_tensor
    
    def apply_pitch_shift(self, waveform_np: np.ndarray, sampling_rate:int) -> np.ndarray:
        if random.random() < self.pitch_shift_prob:
            n_steps = random.uniform(self.pitch_shift_semitones_min, self.pitch_shift_semitones_max)
            try:
                return librosa.effects.pitch_shift(y=waveform_np, sr=sampling_rate, n_steps=n_steps)
            except Exception as e:
                print(f"Warning: Could not apply pitch shift: {e}")
        return waveform_np
    
    def apply_tempo_perturb(self, waveform_np: np.ndarray) -> np.ndarray:
        if random.random() < self.tempo_perturb_prob:
            rate = random.uniform(self.tempo_rate_min, self.tempo_rate_max)
            try:
                # librosa.effects.time_stretch can be slow for long audio.
                # For very long audio, consider torchaudio.transforms.SpeedPerturbation (if Kaldi available)
                # or sox effects for speed. For typical segment lengths, librosa is fine.
                return librosa.effects.time_stretch(y=waveform_np, rate=rate)
            except Exception as e:
                print(f"Warning: Could not apply tempo perturbation: {e}")
        return waveform_np
  
    def add_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(waveform) * self.noise_level
        return waveform + noise
    
    def spec_augment(self, mel_spec, time_mask_param=30, freq_mask_param=13, prob=0.2):
        if random.random() > prob:
            return mel_spec  # no masking

        # Apply 2 masks each
        for _ in range(2):
            mel_spec = T.TimeMasking(time_mask_param=time_mask_param)(mel_spec)
            mel_spec = T.FrequencyMasking(freq_mask_param=freq_mask_param)(mel_spec)

        return mel_spec
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # print("Batch keys:", features[0].keys())
        if "input_features" not in features[0]:
            if random.random() < 0.5:
                input_features = []
                for feature in features:
                    waveform_np = feature["audio"]["array"].astype(np.float32)
                    sampling_rate = feature["audio"]["sampling_rate"]
                    
                    # Order of NumPy-based augmentations:
                    # 1. Tempo Perturbation
                    waveform_np = self.apply_tempo_perturb(waveform_np)
                    # 2. Pitch Shifting
                    waveform_np = self.apply_pitch_shift(waveform_np, sampling_rate)
                    
                    # Convert np to tensor for torchaudio/tensor-based augmentations
                    waveform_tensor = torch.from_numpy(waveform_np.copy()) # Use .copy() if waveform_np is modified by tensor ops
                    
                    # Tensor-based augmentations:
                    # 3. Apply reverberation
                    # waveform_tensor = self.apply_reverb(waveform_tensor, sampling_rate)
                    # 4. Add Gaussian noise
                    waveform_tensor = self.add_noise(waveform_tensor)

                    # Re-extract features from noisy waveform
                    augmented_audio_np = waveform_tensor.numpy()
                    inputs = self.processor.feature_extractor(
                        augmented_audio_np, sampling_rate=sampling_rate, return_tensors="pt"
                    )
                    augmented_inputs = self.spec_augment(inputs.input_features[0])
                    # input_features.append({"input_features": inputs["input_features"][0]})
                    input_features.append({"input_features": augmented_inputs})
            else:
                input_features = []
                for feature in features:
                    waveform_np = feature["audio"]["array"]
                    sampling_rate = feature["audio"]["sampling_rate"]
                    inputs = self.processor.feature_extractor(
                        waveform_np, sampling_rate=sampling_rate, return_tensors="pt"
                    )
                    input_features.append({"input_features": inputs["input_features"][0]})

        else:
            # Evaluation mode — use precomputed features
            input_features = [{"input_features": f["input_features"]} for f in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenised label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's appended later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return {
            "input_features": batch["input_features"],
            "labels": batch["labels"],
        }

data_collator = DataCollatorSpeechSeq2SeqWithPaddingAndNoise(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,

    # --- Configure which augmentations to use ---
    # Reverb
    reverb_prob=REVERB_PROB,
    reverb_profiles=REVERB_PROFILE,

    # Pitch Shifting
    pitch_shift_prob=PITCH_SHIFT_PROB,
    pitch_shift_semitones_min=PITCH_SHIFT_SEMITONES_MIN,
    pitch_shift_semitones_max=PITCH_SHIFT_SEMITONES_MAX,

    # Tempo Perturbation
    tempo_perturb_prob=TEMPO_PERTURB_PROB,
    tempo_rate_min=TEMPO_RATE_MIN,
    tempo_rate_max=TEMPO_RATE_MAX,

    # Gaussian Noise
    noise_prob=NOISE_PROB,
    noise_level=NOISE_LEVEL
)

# data_collator = DataCollatorSpeechSeq2SeqWithAugment(processor=processor, decoder_start_token_id=model.config.decoder_start_token_id)

# def debug_collator(batch):
#     result = data_collator(batch)
#     result = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in result.items()}
#     print(result["input_features"].device)  # Should print cuda:0 during training
#     return result

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'-', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text.strip())
    return text

metric = evaluate.load("wer")
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    pred_str_normalised = [preprocess_text(text) for text in pred_str]
    label_str_normalised = [preprocess_text(text) for text in label_str]
    
    wer = 100 * metric.compute(predictions=pred_str_normalised, references=label_str_normalised)

    return {"wer": wer}


training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper_small_finetuned_model2",  # change to a repo name of your choice
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    logging_dir="./whisper_small_logs2",
    learning_rate=5e-6,
    # warmup_steps=1000,
    warmup_ratio=0.2,
    num_train_epochs=20,
    # max_steps=200,
    gradient_checkpointing=True,
    fp16=True,
    per_device_eval_batch_size=2,
    weight_decay=0.01,
    predict_with_generate=True,
    generation_max_length=160,  
    save_strategy="steps",
    lr_scheduler_type='cosine',
    save_steps=1500,
    eval_strategy="steps",
    eval_steps=1500,
    logging_steps=25,
    save_total_limit=2, 
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    remove_unused_columns=False,
    adam_beta1=0.9,
    adam_beta2=0.999,
    dataloader_num_workers=16,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
)

# print(training_args.dataloader_num_workers)

# print(next(model.parameters()).device)
# trainer.train(resume_from_checkpoint=True)
trainer.train()
trainer.save_model()

predictions_aft_finetuning = trainer.predict(test_dataset)
wer_aft_finetuning = compute_metrics(predictions_aft_finetuning)

print(f"WER aft fine-tuning: {wer_aft_finetuning['wer']}")
