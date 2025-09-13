import torch
import random
import numpy as np
from datasets import load_dataset, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback

import torchaudio.transforms as T # For SpecAugment
import evaluate
import jiwer # For WER calculation and normalisation

from dataclasses import dataclass
from typing import Any, Dict, List, Union

# --- Install audiomentations ---
# If you haven't already, make sure to install it:
# pip install audiomentations

# --- Import audiomentations ---
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Gain, PolarityInversion, ClippingDistortion, HighPassFilter, LowPassFilter, Shift, AddColorNoise # AddColorNoise is a good alternative to real noise

# --- Data Loading and Preparation ---
dataset = load_dataset("json", data_files="./asr.jsonl", split="train")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

MODEL_NAME = "openai/whisper-tiny"
feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, language="English", task="transcribe")
processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="English", task="transcribe")

def prepare_train_dataset(batch):
    # encode target text to label ids
    batch["labels"] = tokenizer(batch["transcript"]).input_ids
    return batch

def prepare_test_dataset(batch):
    audio = batch["audio"]
    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    # encode target text to label ids
    batch["labels"] = tokenizer(batch["transcript"]).input_ids
    return batch

split_dataset = dataset.train_test_split(test_size=0.1, seed=42)

# Access splits
train_dataset = split_dataset["train"]
test_dataset = split_dataset["test"]

train_dataset = train_dataset.map(prepare_train_dataset)
test_dataset = test_dataset.map(prepare_test_dataset,remove_columns=test_dataset.column_names)

# Enable grad for model
torch.set_grad_enabled(True)

model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
model.generation_config.language = "english"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None
model.generation_config.use_cache = False
model.gradient_checkpointing_enable()

# --- Data Collator with audiomentations ---
@dataclass
class DataCollatorSpeechSeq2SeqWithPaddingAndAugment:
    processor: WhisperProcessor
    decoder_start_token_id: int
    
    # --- audiomentations pipeline ---
    def __post_init__(self):
        # The audiomentations Compose pipeline
        # Aggressive parameters chosen for when you lack real noise/RIR data
        self.audio_augmenter = Compose([
            # Noise Augmentations (Synthetic)
            # AddGaussianNoise: Simple white noise
            AddGaussianNoise(min_amplitude=0.003, max_amplitude=0.015, p=0.6), # Higher amplitude, higher prob
            # AddColorNoise: Various types of colored noise (Pink, Brown, Blue, Violet)
            # Simulates different real-world noise characteristics without needing files
            AddColorNoise(min_snr_db=5.0, max_snr_db=25.0, p=0.4,
                          min_f_decay=0.0, max_f_decay=2.0), # Random f_decay for different "colors"

            # Basic Speech Augmentations
            Gain(min_gain_db=-10.0, max_gain_db=10.0, p=0.5), # More aggressive volume perturbation
            TimeStretch(min_rate=0.8, max_rate=1.2, p=0.3), # Wider range for tempo perturbation
            PitchShift(min_semitones=-3.5, max_semitones=3.5, p=0.3), # Wider range for pitch shifting
            Shift(min_shift=-0.2, max_shift=0.2, p=0.15), # Larger time shift range

            # Channel/Quality Degradation (simulate real-world audio quality)
            # These are crucial when you don't have real noise/RIRs
            # ClippingDistortion(min_percentile_threshold=0.01, max_percentile_threshold=8.0, p=0.1), # More aggressive clipping
            LowPassFilter(min_cutoff_freq=2000, max_cutoff_freq=7000, p=0.15), # More aggressive low-pass filtering (e.g., phone calls)
            HighPassFilter(min_cutoff_freq=80, max_cutoff_freq=300, p=0.1), # More aggressive high-pass filtering
            PolarityInversion(p=0.05), # Invert audio polarity
        ])

        # SpecAugment parameters (using torchaudio as before, or audiomentations.transforms.spectrogram)
        self.spec_augment_prob: float = 0.7 # Increased probability
        self.time_mask_param_range: tuple = (25, 50) # Wider range for time mask
        self.freq_mask_param_range: tuple = (12, 18) # Wider range for freq mask

    def spec_augment(self, mel_spec: torch.Tensor) -> torch.Tensor:
        if random.random() < self.spec_augment_prob:
            time_mask_param = random.randint(*self.time_mask_param_range)
            freq_mask_param = random.randint(*self.freq_mask_param_range)
            
            for _ in range(2): # Apply 2 masks each
                mel_spec = T.TimeMasking(time_mask_param=time_mask_param)(mel_spec)
                mel_spec = T.FrequencyMasking(freq_mask_param=freq_mask_param)(mel_spec)
        return mel_spec
    
    def __call__(self, features: List[Dict[str, Union[List[int], np.ndarray]]]) -> Dict[str, torch.Tensor]:
        if "input_features" not in features[0]: # This means it's a training batch
            input_features = []
            for feature in features:
                audio_array = feature["audio"]["array"].astype(np.float32) # Ensure float32 for audiomentations
                sampling_rate = feature["audio"]["sampling_rate"]
                
                # Apply audio-level augmentations
                try:
                    augmented_audio_array = self.audio_augmenter(samples=audio_array, sample_rate=sampling_rate)
                except Exception as e:
                    print(f"Error during audiomentation for a sample: {e}. Using original audio.")
                    augmented_audio_array = audio_array # Fallback to original

                # Convert augmented NumPy array to Mel-spectrogram
                inputs = self.processor.feature_extractor(
                    augmented_audio_array, sampling_rate=sampling_rate, return_tensors="pt"
                )
                
                # Apply SpecAugment to the Mel-spectrogram
                augmented_mel_spec = self.spec_augment(inputs.input_features.squeeze(0)) # Remove batch dim

                input_features.append({"input_features": augmented_mel_spec})
        else:
            # Evaluation mode: use precomputed features, no augmentation
            input_features = [{"input_features": f["input_features"]} for f in features]

        # Pad input features (Mel-spectrograms)
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Get and pad tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # If BOS token is appended in previous tokenization step, cut BOS token here
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return {
            "input_features": batch["input_features"],
            "labels": batch["labels"],
        }

# --- Initialize Data Collator ---
data_collator = DataCollatorSpeechSeq2SeqWithPaddingAndAugment(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

# --- WER Metric Setup ---
metric = evaluate.load("wer")
jiwer_transform = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.SubstituteRegexes({r"-": r" "}),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.RemoveWhiteSpace(replace_by_space=True),
    jiwer.Strip(),
])

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    pred_str_normalised = [jiwer_transform(s) for s in pred_str]
    label_str_normalised = [jiwer_transform(s) for s in label_str]
    
    wer_score_fraction = metric.compute(predictions=pred_str_normalised, references=label_str_normalised)
    
    return {"wer": wer_score_fraction * 100}

# --- Training Arguments ---
training_args = Seq2SeqTrainingArguments(
    output_dir="./tiny_augmented", # Changed output directory name
    per_device_train_batch_size=32,
    gradient_accumulation_steps=1,
    logging_dir="./logs_augmented", # Changed logging directory name
    learning_rate=1e-5,
    warmup_ratio=0.1,
    num_train_epochs=15, # Increased epochs for better convergence with more aggressive augmentation
    gradient_checkpointing=True,
    fp16=True,
    per_device_eval_batch_size=8,
    save_strategy="steps",
    save_steps=500,
    eval_strategy="steps",
    eval_steps=100,
    logging_steps=25,
    save_total_limit=2,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    remove_unused_columns=False,
    label_names=["labels"]
)

early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)

# --- Trainer Setup ---
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
    callbacks=[early_stopping_callback]
)

# --- Start Training ---
print("Starting training...")
trainer.train()

# --- Save Model ---
trainer.save_model("tiny_augmented_no_external_data")
tokenizer.save_pretrained("tiny_augmented_no_external_data")

print("Training complete")