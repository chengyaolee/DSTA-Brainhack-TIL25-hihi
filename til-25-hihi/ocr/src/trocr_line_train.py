import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os # Added for path joining
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DefaultDataCollator
from evaluate import load as load_metric # For evaluation

# --- Configuration for Preprocessed Data ---
# This should match the OUTPUT_RECOGNITION_DATA_DIR in your preprocessing script
PREPROCESSED_DATA_DIR = "./paddleocr_rec_data_prepared2" # ADJUST THIS PATH if different

class OCRDataset(Dataset):
    def __init__(self, df, processor, max_target_length=128):
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df.iloc[idx]['image_path']
        text = self.df.iloc[idx]['text']

        try:
            image = Image.open(image_path).convert("RGB")
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze()
        except FileNotFoundError:
            print(f"Warning: Image not found at {image_path}, skipping this item.")
            # A robust way to handle this is to return None and filter in collate_fn
            # or ensure all paths in the DataFrame are valid before creating the Dataset.
            # For simplicity, if an error occurs, we'll try to get the next item.
            # This recursive call needs to be handled carefully to avoid RecursionError for many missing files.
            if idx + 1 < len(self.df):
                print(f"Attempting to load next item: {idx + 1}")
                return self.__getitem__(idx + 1)
            else:
                # If it's the last item or recursion fails, raise error or return a marker
                raise FileNotFoundError(f"Critical: Image not found at {image_path} and no more items to try.")
        except Exception as e:
            print(f"Warning: Error loading image {image_path}: {e}. Skipping this item.")
            if idx + 1 < len(self.df):
                return self.__getitem__(idx + 1)
            else:
                raise RuntimeError(f"Critical: Error loading image {image_path} and no more items: {e}")


        labels = self.processor.tokenizer(text,
                                          padding="max_length",
                                          max_length=self.max_target_length,
                                          truncation=True).input_ids
        
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
        labels = torch.tensor(labels)

        return {"pixel_values": pixel_values, "labels": labels}

# --- Function to Load Data from Preprocessing Output ---
def load_custom_ocr_data(label_file_path, base_image_dir):
    """
    Loads OCR dataset from a label file and constructs absolute image paths.

    Args:
        label_file_path (str): Absolute path to the 'train_rec_label.txt' or 'val_rec_label.txt'.
        base_image_dir (str): The root directory of the preprocessed data
                              (e.g., PREPROCESSED_DATA_DIR).
                              Relative paths in the label file are joined with this.
    Returns:
        pd.DataFrame: DataFrame with 'image_path' (absolute) and 'text' columns.
    """
    image_paths = []
    texts = []
    if not os.path.exists(label_file_path):
        print(f"Error: Label file not found: {label_file_path}")
        return pd.DataFrame({'image_path': [], 'text': []})

    with open(label_file_path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line: # Skip empty lines
                continue
            parts = line.split('\t', 1)
            if len(parts) == 2:
                relative_img_path, text_content = parts
                # Image paths in label file are relative to PREPROCESSED_DATA_DIR
                # e.g., "train_crops/image_001.png"
                abs_img_path = os.path.join(base_image_dir, relative_img_path)
                
                # Basic check for text content, can be expanded
                if not text_content:
                    # print(f"Warning: Empty text for image {relative_img_path} in {label_file_path}, line {line_idx+1}. Skipping.")
                    continue # Skip if text is empty

                image_paths.append(abs_img_path)
                texts.append(text_content)
            else:
                print(f"Warning: Malformed line in {label_file_path} (line {line_idx+1}): '{line}'. Skipping.")
                
    if not image_paths:
        print(f"Warning: No valid data loaded from {label_file_path}.")
        
    return pd.DataFrame({'image_path': image_paths, 'text': texts})

if __name__ == "__main__":
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-printed', use_fast=False)
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-printed')

    # Set decoder_start_token_id and other special tokens for generation
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # Set beam search parameters
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 128 # Adjusted to be more reasonable for line-level OCR and match max_target_length
                                 # Original was 8192, which is very large for single lines.
                                 # This affects predict_with_generate.
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 1.0 # Changed from 2.0 to 1.0 as a common default
    model.config.num_beams = 4

    # 1. Load your metadata using the new data loading function
    train_label_file = os.path.join(PREPROCESSED_DATA_DIR, "train_rec_label.txt")
    eval_label_file = os.path.join(PREPROCESSED_DATA_DIR, "val_rec_label.txt")

    print(f"Attempting to load training data from: {train_label_file}")
    train_df = load_custom_ocr_data(train_label_file, PREPROCESSED_DATA_DIR)
    print(f"Attempting to load evaluation data from: {eval_label_file}")
    eval_df = load_custom_ocr_data(eval_label_file, PREPROCESSED_DATA_DIR)

    if train_df.empty:
        print(f"Critical Error: Training data is empty. Please check {train_label_file} and image paths.")
        exit()
    if eval_df.empty:
        print(f"Warning: Evaluation data is empty. Please check {eval_label_file} and image paths. Evaluation steps might fail.")
        # You might want to exit() here too if evaluation is mandatory for your workflow

    print(f"Loaded {len(train_df)} training samples and {len(eval_df)} evaluation samples.")
    if not train_df.empty:
        print("Sample training data point:")
        print(f"  Image Path: {train_df.iloc[0]['image_path']}")
        print(f"  Text: {train_df.iloc[0]['text']}")
    if not eval_df.empty:
        print("Sample evaluation data point:")
        print(f"  Image Path: {eval_df.iloc[0]['image_path']}")
        print(f"  Text: {eval_df.iloc[0]['text']}")


    # 2. Create Dataset instances
    # max_target_length for OCRDataset should be chosen based on your data's text line lengths.
    # It's the max number of tokens after tokenization.
    # This value is also used in model.config.max_length for consistency during generation.
    dataset_max_target_length = 128
    train_dataset = OCRDataset(df=train_df, processor=processor, max_target_length=dataset_max_target_length)
    eval_dataset = OCRDataset(df=eval_df, processor=processor, max_target_length=dataset_max_target_length)
    
    # Filter out None items from dataset if __getitem__ can return None
    # This is a more robust way to handle it if many FileNotFoundError are expected
    # and you prefer filtering at the DataLoader level.
    # However, the current __getitem__ tries to recover or raises an error.

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        eval_strategy="epoch",       # Use "evaluation_strategy" for newer Transformers versions
        save_strategy="epoch",       # Use "save_strategy" for newer Transformers versions
        per_device_train_batch_size=4, # Adjust based on your GPU memory
        per_device_eval_batch_size=4,  # Adjust based on your GPU memory
        fp16=torch.cuda.is_available(),
        output_dir="./trocr_finetuned_run/", # Changed to avoid conflict if rerunning
        logging_steps=50, # Increased logging steps
        num_train_epochs=5, # Example: Increased epochs, adjust as needed
        learning_rate=5e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        # report_to="wandb", # Optional
    )

    # Data collator
    data_collator = DefaultDataCollator()

    # Evaluation metric (Word Error Rate - WER)
    wer_metric = load_metric("wer")
    # You could also use CER (Character Error Rate)
    # cer_metric = load_metric("cer")

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        # Replace -100 in labels as we can't decode them
        labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id

        # Decode predictions and labels
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)
        
        # Simple cleaning for WER calculation:
        # Optional: normalize strings (e.g., lowercase, remove punctuation) 
        # if your task benefits from it and the metric should reflect that.
        # pred_str_cleaned = [s.lower().strip() for s in pred_str]
        # label_str_cleaned = [s.lower().strip() for s in label_str]

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        # cer = cer_metric.compute(predictions=pred_str, references=label_str) # If you also want CER
        
        return {"wer": wer} #, "cer": cer}

    # Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.tokenizer, # ✅ Corrected: Should be the text tokenizer part of the processor
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # GPU check
    if torch.cuda.is_available():
        print("CUDA is available. Training on GPU.")
        # model.to('cuda') # Trainer handles device placement
    else:
        print("CUDA not available. Training on CPU (will be very slow).")

    try:
        print("Starting training...")
        trainer.train()
    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging

    # Save the final model and processor
    final_model_path = "./trocr_finetuned_final_model/"
    model.save_pretrained(final_model_path)
    processor.save_pretrained(final_model_path)
    print(f"Training complete. Model and processor saved to {final_model_path}")

    # Evaluate the model (optional, if you want a final evaluation printout)
    if eval_dataset:
        print("Evaluating the final model...")
        eval_results = trainer.evaluate()
        print("Final evaluation results:")
        for key, value in eval_results.items():
            print(f"  {key}: {value}")