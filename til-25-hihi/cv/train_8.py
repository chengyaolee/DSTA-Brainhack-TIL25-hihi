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
            # Multi-scale training augmentations - Critical for small objects
            A.RandomScale(scale_limit=(-0.2, 0.5), p=0.7),  # More upscaling for small objects
            A.LongestMaxSize(max_size=int(IMAGE_SIZE * 1.4), p=0.4),  # Larger resize range
            A.SmallestMaxSize(max_size=int(IMAGE_SIZE * 0.6), p=0.4),  # More aggressive small size
            
            # Smart cropping that preserves small objects
            A.RandomSizedBBoxSafeCrop(
                height=int(0.8 * IMAGE_SIZE), 
                width=int(0.8 * IMAGE_SIZE),
                erosion_rate=0.02,  # Reduced erosion to preserve small objects
                p=0.3
            ),
            # Crop and pad strategy for maintaining object visibility
            A.CropAndPad(
                percent=(-0.1, 0.1), 
                pad_mode=cv2.BORDER_CONSTANT, 
                pad_cval=0, 
                p=0.3
            ),
            
            # Advanced mixing augmentations for better generalization
            A.MixUp(alpha=0.3, p=0.2),  # Slightly stronger mixing
            A.CutMix(alpha=1.2, p=0.2),  # Better for small object learning
            
            # Minimal noise since training images are clean
            A.GaussNoise(var_limit=(3, 8), p=0.15),  # Light noise for robustness
            A.ISONoise(color_shift=(0.005, 0.015), intensity=(0.1, 0.2), p=0.1),
            
            # Geometric transformations optimized for detection
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.15),  # Some vehicle classes benefit from this
            A.RandomRotate90(p=0.25),
            A.ShiftScaleRotate(
                shift_limit=0.08, 
                scale_limit=0.15, 
                rotate_limit=12, 
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.4
            ),
            # Perspective changes for better generalization
            A.Perspective(scale=(0.05, 0.1), p=0.3),
            
            # Enhanced contrast and visibility for small objects
            A.RandomBrightnessContrast(
                brightness_limit=0.2, 
                contrast_limit=0.25, 
                p=0.4
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.3),  # Better local contrast
            A.Equalize(mode='cv', by_channels=True, p=0.15),  # Adaptive histogram equalization
            
            # Minimal blur to avoid destroying small objects
            A.Blur(blur_limit=2, p=0.02),  # Very minimal blur
            A.GaussianBlur(blur_limit=2, p=0.02),
            
            # Color augmentations for robustness
            A.HueSaturationValue(
                hue_shift_limit=12, 
                sat_shift_limit=20, 
                val_shift_limit=12, 
                p=0.4
            ),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.3),
            A.ToGray(p=0.03),  # Minimal grayscale
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
            
            # Quality and compression
            A.ImageCompression(quality_lower=80, quality_upper=100, p=0.2),
            A.Downscale(scale_min=0.85, scale_max=0.95, p=0.15),  # Simulate low-res images
            
            # Padding (ensure consistent size)
            A.PadIfNeeded(IMAGE_SIZE, IMAGE_SIZE, cv2.BORDER_CONSTANT, [0,0,0], p=1.0)
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
        epochs=150,
        batch=20,
        imgsz=IMAGE_SIZE,

        optimizer="AdamW",  # Better for small object detection
        lr0=0.001,   # Lower initial learning rate
        lrf=0.01,    # Final learning rate factor
        momentum=0.9,
        weight_decay=0.001,  # Increased regularization

        # Augmentation parameters optimized for small objects
        mosaic=0.8,    # Reduced slightly to preserve small objects
        mixup=0.1,     # Enable mixup
        flipud=0.15,   # Vertical flip probability
        fliplr=0.5,    # Horizontal flip probability
        degrees=10.0,  # Reduced rotation
        translate=0.1, # Translation fraction
        scale=0.3,     # Scale factor
        shear=5.0,     # Shear degrees

        # Loss weights optimized for small objects
        cls=0.5,       # Classification loss weight
        box=7.5,       # Box regression loss weight (increased for small objects)
        dfl=1.5,       # DFL loss weight

        # Training behavior
        patience=30,   # Increased early stopping patience
        save_period=5, # Save checkpoint every 5 epochs
        workers=8,     # Data loading workers
        seed=42,       # Reproducibility

        # Disable label smoothing for small objects
        label_smoothing=0.0,

        # Other settings
        name='yolo11_v8',
        save=True,
        plots=True,
        val=True,

        # Enable augmentation
        augment=True,
    )
    
    # Save the trained model
    model_save_path = output_path / "best_v8.pt"
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
