import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
from jiwer import wer
from bs4 import BeautifulSoup

# Load your dataset
df = pd.read_csv("labels.csv")
df = df.head(2)

# Device setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model and processor
processor = AutoProcessor.from_pretrained("ds4sd/SmolDocling-256M-preview")
model = AutoModelForVision2Seq.from_pretrained(
    "ds4sd/SmolDocling-256M-preview",
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    _attn_implementation="eager"
).to(DEVICE)

def extract_raw_text(docling_output):
    soup = BeautifulSoup(docling_output, "xml")
    return soup.get_text(separator=" ").strip()

def generate_from_batch(image_batch, batch_prompts):
    inputs = processor(text=batch_prompts, images=image_batch, return_tensors="pt", padding=True).to(DEVICE)
    generated_ids = model.generate(**inputs, max_new_tokens=8192)
    prompt_len = inputs.input_ids.shape[1]
    trimmed_ids = generated_ids[:, prompt_len:]
    outputs = processor.batch_decode(trimmed_ids, skip_special_tokens=False)
    return [extract_raw_text(out.lstrip()) for out in outputs]

# Batched inference
batch_size = 2  # Adjust based on GPU memory
predictions = []
messages_template = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Convert this page to docling."}
        ]
    },
]

for i in tqdm(range(0, len(df), batch_size)):
    batch_df = df.iloc[i:i+batch_size]
    image_paths = batch_df["image_path"].tolist()

    # Load images safely
    images = []
    valid_indices = []
    for idx, path in enumerate(image_paths):
        try:
            images.append(load_image(path))
            valid_indices.append(idx)
        except Exception as e:
            print(f"[ERROR loading {path}]: {e}")
            images.append(None)

    # Filter out any failed images
    valid_images = [img for img in images if img is not None]
    if not valid_images:
        predictions.extend([""] * len(image_paths))
        continue

    # Generate prompts
    prompts = [processor.apply_chat_template(messages_template, add_generation_prompt=True)] * len(valid_images)
    
    try:
        outputs = generate_from_batch(valid_images, prompts)
        # Reinsert predictions in original order (handling failed loads)
        output_iter = iter(outputs)
        batch_predictions = [next(output_iter) if img is not None else "" for img in images]
    except Exception as e:
        print(f"[ERROR in batch {i}]: {e}")
        batch_predictions = [""] * len(image_paths)

    predictions.extend(batch_predictions)

# Add predictions and evaluate WER
print(df.loc[0]['text'])
print(predictions[0])
df["predicted_text"] = predictions
overall_wer = wer(df["text"].tolist(), df["predicted_text"].tolist())
print(f"\n✅ Batched Overall WER: {overall_wer:.4f}")

# (Optional) Save results
# df.to_csv("predictions_with_wer.csv", index=False)
