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
    """Initialize transform with YOLO bbox handling."""
    import albumentations as A
    import cv2
    from ultralytics.utils import LOGGER, colorstr

    self.p = p
    self.transform = None
    prefix = colorstr("albumentations: ")

    try:
        T = [
            # --- Scaling and Cropping for Small Objects ---
            # Increase the chance of scaling images up to make small objects larger.
            A.RandomScale(scale_limit=0.5, p=0.5),
            A.RandomSizedBBoxSafeCrop(
                height=int(0.8 * IMAGE_SIZE),
                width=int(0.8 * IMAGE_SIZE),
                erosion_rate=0.02, # Keep erosion low to avoid clipping small objects.
                p=0.4
            ),
            # Pad back to the required size. This must be the last spatial transform.
            A.PadIfNeeded(IMAGE_SIZE, IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT, value=[0,0,0], p=1.0),

            # --- Augmentations for Noisy Images ---
            # A balanced mix of noise, blur, and quality degradation to build robustness.
            A.GaussNoise(var_limit=(20, 50), p=0.3),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=0.2),
            A.ImageCompression(quality_lower=75, quality_upper=95, p=0.3),
            A.Downscale(scale_min=0.75, scale_max=0.95, p=0.2),
            # Use mild blur, as heavy blur can destroy small object details.
            A.MotionBlur(blur_limit=5, p=0.1),
            A.Blur(blur_limit=3, p=0.1),

            # --- Color, Contrast, and Lighting ---
            # Helps the model generalize to different conditions.
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.CLAHE(clip_limit=3.0, p=0.2), # Good for enhancing local contrast.
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.4),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.2),

            # --- Geometric Transformations ---
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1), # Only if relevant for your dataset.
            A.RandomRotate90(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.15, rotate_limit=15, p=0.3),
        ]

        self.bbox_transform = A.Compose(T, bbox_params=A.BboxParams('yolo', ['class_labels']))
        self.image_transform = A.Compose([t for t in T if not isinstance(t, A.RandomSizedBBoxSafeCrop)])
        self.transform = self.adaptive_transform
        self.contains_spatial = True

        LOGGER.info(prefix + ", ".join(f"{x}" for x in T if hasattr(x, 'p') and x.p > 0))
    except Exception as e:
        LOGGER.warning(f"{prefix} Failed: {e}")
        self.transform = lambda **kwargs: kwargs

def adaptive_transform(self, **kwargs):
    """Transform with YOLO data structure handling."""
    if "image" not in kwargs:
        return kwargs
    
    image = kwargs["image"]
    bboxes = kwargs.get("bboxes")
    labels = kwargs.get("class_labels") or kwargs.get("labels")
    
    try:
        # Safe length check for arrays/tensors
        bbox_len = len(bboxes) if bboxes is not None else 0
        if bboxes is not None and labels is not None and bbox_len > 0:
            # Ensure proper format
            if hasattr(bboxes, 'tolist'):
                bboxes = bboxes.tolist()
            if hasattr(labels, 'tolist'):
                labels = labels.tolist()
            
            # Flatten labels if nested
            if labels and hasattr(labels[0], '__len__'):
                try:
                    if len(labels[0]) == 1:
                        labels = [item[0] if hasattr(item, '__getitem__') else item for item in labels]
                except (TypeError, IndexError):
                    pass
            
            result = self.bbox_transform(image=image, bboxes=bboxes, class_labels=labels)
            return {
                "image": result["image"],
                **{k: result.get(k, v) for k, v in kwargs.items() 
                   if k in ['bboxes', 'class_labels', 'labels'] and k in result}
            }
        else:
            return {"image": self.image_transform(image=image)["image"]}
    except Exception as e:
        print(f"Adaptive Transform failed: {e}")
        return {"image": image}

def enhanced_call(self, labels, image=None):
    """Enhanced call method for YOLO data extraction."""
    # Get image - avoid ambiguous array truth values
    img = None
    if image is not None:
        img = image
    elif 'img' in labels and labels['img'] is not None:
        img = labels['img']
    elif 'image' in labels and labels['image'] is not None:
        img = labels['image']
    
    if img is None:
        return labels
    
    # Extract bboxes and labels from instances
    bboxes = class_labels = None
    if 'instances' in labels and labels['instances'] is not None:
        inst = labels['instances']
        if hasattr(inst, 'bboxes') and inst.bboxes is not None:
            bbox_shape = getattr(inst.bboxes, 'shape', None)
            bboxes = inst.bboxes.tolist() if bbox_shape and len(bbox_shape) == 2 else None
        if hasattr(inst, 'cls') and inst.cls is not None:
            class_labels = inst.cls.tolist()
    
    # Fallback to direct cls field
    if class_labels is None and 'cls' in labels and labels['cls'] is not None:
        cls_data = labels['cls']
        class_labels = (cls_data.cpu().numpy().flatten().tolist() if hasattr(cls_data, 'cpu')
                       else cls_data.flatten().tolist() if hasattr(cls_data, 'flatten')
                       else cls_data)
    
    # Apply transform
    if hasattr(self, 'transform') and callable(self.transform):
        try:
            transform_args = {"image": img}
            if bboxes is not None and class_labels is not None:
                transform_args.update({"bboxes": bboxes, "class_labels": class_labels})
            
            result = self.transform(**transform_args)
            
            # Update results
            if image is not None:
                image = result['image']
            else:
                labels['img'] = result['image']
            
            # Update instances if transformed - handle read-only properties
            if 'bboxes' in result and 'instances' in labels and labels['instances']:
                import torch
                try:
                    # Try direct assignment first
                    new_bboxes = torch.tensor(result['bboxes'], dtype=torch.float32,
                                            device=labels['instances'].bboxes.device)
                    labels['instances'].bboxes = new_bboxes
                except (AttributeError, RuntimeError):
                    # If bboxes is read-only, create new Instances object
                    from ultralytics.utils.instance import Instances
                    new_instances = Instances(
                        bboxes=torch.tensor(result['bboxes'], dtype=torch.float32,
                                          device=labels['instances'].bboxes.device),
                        segments=getattr(labels['instances'], 'segments', None),
                        keypoints=getattr(labels['instances'], 'keypoints', None),
                        bbox_format=getattr(labels['instances'], 'bbox_format', 'xywh'),
                        normalized=getattr(labels['instances'], 'normalized', True)
                    )
                    if hasattr(labels['instances'], 'cls'):
                        new_instances.cls = labels['instances'].cls
                    labels['instances'] = new_instances
            
            if 'class_labels' in result and 'instances' in labels and labels['instances']:
                import torch
                try:
                    labels['instances'].cls = torch.tensor(
                        result['class_labels'], dtype=torch.long,
                        device=labels['instances'].cls.device if hasattr(labels['instances'], 'cls') 
                        else labels['instances'].bboxes.device)
                except (AttributeError, RuntimeError):
                    # cls might also be read-only in some versions
                    pass
                    
        except Exception as e:
            print(f"Transform failed: {e}")
    
    return labels

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
    model = YOLO("yolo11l.pt")
    
    # Train the model
    # Now we can use 'albumentations' parameter because we patched the Albumentations class

    results = model.train(
        data=str(output_path / "dataset.yaml"),
        augment=False,

        # Training Configuration
        epochs=150,           # Train longer for better convergence.
        batch=32,
        imgsz=IMAGE_SIZE,
        patience=30,

        # Optimizer and Hyperparameters
        optimizer="AdamW",    # Generally performs well.
        lr0=0.001,            # Initial learning rate.
        lrf=0.01,             # Final learning rate factor.
        weight_decay=0.0005,  # Standard weight decay.
        label_smoothing=0.0,  # Turn off for more precise bounding boxes on small objects.

        # Loss Function Weights
        # Crucial for focusing the model on what matters.
        box=7.5,              # Increase weight for bounding box loss to prioritize localization.
        cls=0.5,              # Standard classification loss.
        dfl=1.5,              # Standard Distribution Focal Loss.

        # Project Naming
        name='yolo11l_v9',
        save=True,
        save_period=10,
        plots=True,
        val=True,
    )
    
    # Save the trained model
    model_save_path = output_path / "best_v9.pt"
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
