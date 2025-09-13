import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd # If using a CSV for labels

class OCRDataset(Dataset):
    def __init__(self, df, processor, max_target_length=128): # df is your pandas DataFrame from labels.csv
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get image path and text
        image_path = self.df.iloc[idx]['image_path']
        text = self.df.iloc[idx]['text']

        try:
            image = Image.open(image_path).convert("RGB")
            # Prepare image for the model
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze()
        except FileNotFoundError:
            print(f"Warning: Image not found at {image_path}, skipping this item.")
            # Return a dummy item or handle appropriately if you expect missing images
            # For simplicity, we might skip or return None and filter in collate_fn
            # Here, let's try to return the next valid item if possible, or raise error
            # This part needs careful handling based on your dataset's integrity
            if idx + 1 < len(self.df):
                return self.__getitem__(idx + 1) # Be cautious with recursion depth
            else:
                raise # Or return a specific marker

        # Prepare labels for the model (text tokenization)
        labels = self.processor.tokenizer(text,
                                          padding="max_length",
                                          max_length=self.max_target_length,
                                          truncation=True).input_ids
        # Important: Convert tokens to tensor and ensure -100 for padding tokens
        # so they are ignored in the loss computation
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
        labels = torch.tensor(labels)

        return {"pixel_values": pixel_values, "labels": labels}
    
if __name__ == "__main__":
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-printed', use_fast=False) # use_fast=False can help avoid some tokenizer issues
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-printed')

    # Set decoder_start_token_id and other special tokens for generation
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    # make sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size

    # Set beam search parameters (optional, for generation during evaluation if needed)
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 8192 # Adjust based on your expected text length
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 1.0
    model.config.num_beams = 4
    
    from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DefaultDataCollator
    from evaluate import load as load_metric # For evaluation

    # 1. Load your metadata (e.g., labels.csv)
    # Make sure your CSV has 'image_path' and 'text' columns
    try:
        # Example: Assuming 'labels.csv' is in the same directory as your script
        # and image paths in the CSV are absolute or relative to where the script can find them.
        df = pd.read_csv("/home/jupyter/til-25-hihi/ocr/labels.csv") # ADJUST PATH TO YOUR LABELS FILE
        # Create dummy data if labels.csv does not exist, for illustration
    except FileNotFoundError:
        print("labels.csv not found. Creating dummy data for demonstration.")
        exit()

    # 2. Split your dataframe into train and eval
    train_df = df.sample(frac=0.8, random_state=42) # 80% for training
    eval_df = df.drop(train_df.index)               # Remaining 20% for evaluation

    # 3. Create Dataset instances
    train_dataset = OCRDataset(df=train_df, processor=processor)
    eval_dataset = OCRDataset(df=eval_df, processor=processor)

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        eval_strategy="epoch",        # ✅ REQUIRED for load_best_model_at_end
        save_strategy="epoch",              # Matches evaluation strategy
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        fp16=torch.cuda.is_available(),
        output_dir="./trocr_finetuned/",
        logging_steps=10,
        num_train_epochs=3,
        learning_rate=5e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="wer",        # Make sure this matches your compute_metrics output
        greater_is_better=False             # For CER, lower is better
        # report_to="wandb", # Optional
    )

    # Data collator
    data_collator = DefaultDataCollator()

    # Evaluation metric (Character Error Rate - CER)
    wer_metric = load_metric("wer")

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        # Replace -100 in labels as we can't decode them
        labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id

        # Decode predictions and labels
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.feature_extractor, # For image processing part
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Before starting training, ensure your GPU is being utilized if available
    if torch.cuda.is_available():
        print("CUDA is available. Training on GPU.")
        model.to('cuda')
    else:
        print("CUDA not available. Training on CPU (will be very slow).")

    try:
        trainer.train()
    except Exception as e:
        print(f"An error occurred during training: {e}")
        # Potentially log more details or save state if possible

    # Save the final model and processor
    model.save_pretrained("./trocr_finetuned_final/")
    processor.save_pretrained("./trocr_finetuned_final/")
    print("Training complete. Model saved to ./trocr_finetuned_final/")