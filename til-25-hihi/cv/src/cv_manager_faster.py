"""Manages the CV model."""
from typing import Any
import os
import io
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import uuid
import logging as log
import cv2

# --- PyTorch OPTIMIZATION ---
# Set benchmark to True to allow cuDNN to find the best algorithms for your hardware.
# This is beneficial if your input image sizes are consistent.
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

class LetterboxTransform:
    """
    A faster Letterbox transform using OpenCV that resizes and pads the image.
    Pads with a standard grey color (114, 114, 114) to match YOLO training conditions.
    """
    def __init__(self, size=(1280, 1280), color=(114, 114, 114)):
        self.size = size
        self.color = color

    def __call__(self, img_cv: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        """
        Args:
            img_cv (np.ndarray): The input image in OpenCV format (BGR).
        
        Returns:
            A tuple of (padded_image, scale, (pad_left, pad_top)).
        """
        # Resize and pad image while meeting stride-multiple constraints
        shape = img_cv.shape[:2]  # current shape [height, width]
        new_h, new_w = self.size

        # Scale ratio (new / old)
        r = min(new_h / shape[0], new_w / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_w - new_unpad[0], new_h - new_unpad[1]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img_cv = cv2.resize(img_cv, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        # Add border (padding)
        padded_img = cv2.copyMakeBorder(img_cv, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.color)
        
        return padded_img, r, (left, top)


class CVManager:
    def __init__(self):
        yolo_model_path = "best_v7.pt"
        # sahi_model_type = 'yolov8'
        confidence_threshold = 0.15
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # --- MODEL OPTIMIZATION ---
        # Load the standard YOLO model. For max speed, export to FP16 or TensorRT first.
        # Example: model.export(format='torchscript', half=True)
        self.yolo_model = YOLO(yolo_model_path)
        self.yolo_model.to(self.device)

        try:
            # --- SAHI OPTIMIZATION: Load model with half-precision (FP16) ---
            # This is a major speed-up for compatible GPUs.
            self.sahi_detection_model = AutoDetectionModel.from_pretrained(
                model_type=sahi_model_type,
                model_path=yolo_model_path,
                confidence_threshold=confidence_threshold,
                device=self.device,
                # Set half=True for FP16 inference
                half=True if self.device == 'cuda' else False,
            )
            log.info("SAHI model initialized successfully with FP16.")
        except Exception as e:
            log.error(f"Failed to initialize SAHI model: {e}")
            self.sahi_detection_model = None
            
        self.sahi_temp_dir = "temp_sahi_images"
        os.makedirs(self.sahi_temp_dir, exist_ok=True)
        
        # Initialize the faster Letterbox transformer
        self.letterbox = LetterboxTransform(size=(1280, 1280))

    def _preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """
        Applies fast denoising to the image using Gaussian Blur.
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # --- PREPROCESSING OPTIMIZATION: Use GaussianBlur (faster than MedianBlur) ---
        return cv2.GaussianBlur(img_cv, (5, 5), 0)
        
    def cv(self, image: bytes, use_sahi: bool = True) -> list[dict[str, Any]]:
        """Performs object detection on an image."""        
        if use_sahi and self.sahi_detection_model:
            return self._process_with_sahi(image)
        else:
            return self._process_with_yolo(image)

    def _process_with_yolo(self, image: bytes) -> list[dict[str, Any]]:
        """Process image using standard YOLO inference with optimizations."""
        # Preprocess the image (denoise)
        img_cv = self._preprocess_image(image)
        
        # Apply faster letterboxing
        padded_img, scale, padding = self.letterbox(img_cv)
        pad_left, pad_top = padding

        # Convert to tensor and prepare for model
        transform = T.Compose([T.ToTensor()])
        tensor = transform(padded_img).unsqueeze(0)  # Add batch dimension
        
        # --- INFERENCE OPTIMIZATION: Use torch.inference_mode() ---
        with torch.inference_mode():
            result = self.yolo_model.predict(tensor)[0]
        
        boxes = result.boxes
        bounding_boxes_xyxy = boxes.xyxy
        confidence = boxes.conf
        classes = boxes.cls
        
        output = []
        for pred_index in range(len(boxes)):
            x1, y1, x2, y2 = [float(coord) for coord in bounding_boxes_xyxy[pred_index]]
            
            # Adjust for padding and scale to original coordinates
            x1 = (x1 - pad_left) / scale
            y1 = (y1 - pad_top) / scale
            x2 = (x2 - pad_left) / scale
            y2 = (y2 - pad_top) / scale
            
            output.append({
                "bbox": [x1, y1, x2 - x1, y2 - y1], # x, y, w, h
                "category_id": int(classes[pred_index].item()),
                "score": float(confidence[pred_index].item())
            })
            
        return output
    
    def _process_with_sahi(self, image: bytes) -> list[dict[str, Any]]:
        """Process image using SAHI sliced inference with optimizations."""
        if not self.sahi_detection_model:
            log.warning("SAHI model not initialized. Falling back to standard YOLO.")
            return self._process_with_yolo(image)
        
        temp_image_path = None # Ensure variable exists for finally block
        try:
            # Preprocess image (denoise)
            denoised_img_cv = self._preprocess_image(image)

            temp_image_path = os.path.join(self.sahi_temp_dir, f"{uuid.uuid4()}.jpg")
            cv2.imwrite(temp_image_path, denoised_img_cv)
            
            # --- SAHI OPTIMIZATION: Reduced overlap ratio to speed up ---
            result = get_sliced_prediction(
                temp_image_path,
                self.sahi_detection_model,
                slice_height=256,
                slice_width=256,
                overlap_height_ratio=0.1, # Reduced from 0.2
                overlap_width_ratio=0.1,  # Reduced from 0.2
            )
            
            coco_predictions = result.to_coco_predictions(image_id=0)
            
            detections = [{
                "bbox": pred["bbox"],
                "category_id": pred["category_id"],
                "score": pred["score"]
            } for pred in coco_predictions]
            
            return detections
            
        except Exception as e:
            log.error(f"SAHI processing failed: {e}")
            log.warning("Falling back to standard YOLO.")
            return self._process_with_yolo(image)
        finally:
            if temp_image_path and os.path.exists(temp_image_path):
                os.remove(temp_image_path)
