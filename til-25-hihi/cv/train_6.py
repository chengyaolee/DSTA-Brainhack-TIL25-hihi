from pathlib import Path
import os
import json
from ultralytics import YOLO
import cv2
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction, predict
from sahi.utils.file import list_files
import albumentations as A
from ultralytics.data.augment import Albumentations
from ultralytics.utils import LOGGER, colorstr
import albumentations as A

import torch
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

IMAGE_SIZE = 1280

# Monkey patch the Albumentations class to use our custom transforms
def custom_albumentations_init(self, p=1.0):
    """Initialize the transform object for YOLO bbox formatted params."""
    self.p = p
    self.transform = None
    self.contains_spatial = False  # Important to set this to False for non-spatial transforms
    
    prefix = colorstr("albumentations: ")
    try:
        # Define transformations that maintain image dimensions
        T = [
            A.GaussNoise(p=0.4),
            A.ISONoise(p=0.3),
            A.HorizontalFlip(p=0.5),
            A.Blur(p=0.1),
            A.MedianBlur(p=0.1),
            A.ToGray(p=0.1),
            A.CLAHE(p=0.1),
            A.MotionBlur(blur_limit=7, p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.4),
            A.RandomGamma(p=0.2),
            A.ImageCompression(quality_lower=75, p=0.3),
        ]

        self.transform = A.Compose(
            T,
            bbox_params=A.BboxParams(format="yolo")
        )

        LOGGER.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))
    except ImportError:  # package not installed, skip
        pass
    except Exception as e:
        LOGGER.info(f"{prefix}{e}")

# Apply the monkey patch to replace the original __init__ method
Albumentations.__init__ = custom_albumentations_init

def get_dataset_path():
    return Path(os.getcwd()).parent.parent / 'advanced' / 'cv'

def get_output_path():
    # Make sure we write to a directory we have permission to modify
    return Path(os.getcwd()) / "yolo_dataset"

def train():
    # Load YOLO model and train
    output_path = get_output_path()
    output_path.mkdir(exist_ok=True)
    
    # Define metrics output file path
    metrics_file = output_path / "evaluation_metrics.json"
    sahi_metrics_file = output_path / "sahi_evaluation_metrics.json"
    
    # Using standard model name
    model = YOLO("yolo11m.pt")
    
    # Train the model
    # Now we can use 'albumentations' parameter because we patched the Albumentations class
    results = model.train(
        data=str(output_path / "dataset.yaml"),
        epochs=100,
        batch=32,
        imgsz=IMAGE_SIZE,
        name='yolo11_v6',
        save=True,
        save_period=10,
        label_smoothing=0.05,
        patience=10,
        augment=True,
        mosaic=1.0,
        perspective=0.0,
    )
    
    # Save the trained model
    model_save_path = output_path / "best_v6.pt"
    model.save(str(model_save_path))
    
    # Run standard validation and save metrics
    print("Running standard validation...")
    val_results = model.val()
    metrics = {
        "precision": float(val_results.box.map),    # mAP@0.5
        "recall": float(val_results.box.mr),        # mean Average Recall
        "mAP50": float(val_results.box.map50),      # mAP@0.5
        "mAP50-95": float(val_results.box.map),     # mAP@0.5:0.95
        "fitness": float(val_results.fitness)       # Overall fitness score
    }
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Standard evaluation metrics saved to {metrics_file}")
    print(f"Training completed!")
    print(f"Model saved to: {model_save_path}")
    print(f"Training logs saved to: {output_path}/training_run")

if __name__ == "__main__":
    train()
