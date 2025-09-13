"""Manages the CV model."""
from typing import Any
import os
import io
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import uuid
import logging as log

class FastImagePreprocessor:
    """Ultra-fast image preprocessing optimized for speed."""
    
    def __init__(self):
        # Pre-compute kernels for faster operations
        self.sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]], dtype=np.float32)
        
    def fast_denoise(self, img_array: np.ndarray) -> np.ndarray:
        """Ultra-fast denoising using optimized bilateral filter."""
        # Small kernel bilateral filter - much faster than large ones
        return cv2.bilateralFilter(img_array, 5, 50, 50)
    
    def fast_contrast_enhance(self, img_array: np.ndarray, alpha: float = 1.2) -> np.ndarray:
        """Fast contrast enhancement using numpy operations."""
        # Vectorized contrast enhancement - much faster than PIL
        return np.clip(alpha * img_array + (1 - alpha) * 127, 0, 255).astype(np.uint8)
    
    def fast_sharpen(self, img_array: np.ndarray, strength: float = 0.3) -> np.ndarray:
        """Fast sharpening using pre-computed kernel."""
        # Light sharpening - balance between speed and effectiveness
        sharpened = cv2.filter2D(img_array, -1, self.sharpen_kernel * strength)
        return np.clip(img_array + sharpened, 0, 255).astype(np.uint8)
    
    def minimal_preprocessing(self, image: Image.Image) -> Image.Image:
        """Minimal but effective preprocessing - fastest option."""
        img_array = np.array(image)
        
        # Only contrast enhancement - single operation
        enhanced = self.fast_contrast_enhance(img_array, alpha=1.15)
        return Image.fromarray(enhanced)
    
    def balanced_preprocessing(self, image: Image.Image) -> Image.Image:
        """Balanced preprocessing - good speed/quality tradeoff."""
        img_array = np.array(image)
        
        # Two-step process: denoise + contrast
        denoised = self.fast_denoise(img_array)
        enhanced = self.fast_contrast_enhance(denoised, alpha=1.2)
        
        return Image.fromarray(enhanced)
    
    def quality_preprocessing(self, image: Image.Image) -> Image.Image:
        """Higher quality preprocessing - slower but still optimized."""
        img_array = np.array(image)
        
        # Three-step process: denoise + contrast + light sharpen
        denoised = self.fast_denoise(img_array)
        enhanced = self.fast_contrast_enhance(denoised, alpha=1.2)
        sharpened = self.fast_sharpen(enhanced, strength=0.2)
        
        return Image.fromarray(sharpened)

class LetterboxTransform:
    """
    Letterbox transform that resizes and pads the image to maintain aspect ratio.
    Similar to YOLO's letterboxing approach.
    """
    def __init__(self, size=(1280, 1280)):
        self.size = size
        self.target_w, self.target_h = size
        
    def __call__(self, img):
        width, height = img.size
        
        # Pre-calculate to avoid repeated operations
        scale = min(self.target_w / width, self.target_h / height)
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Use LANCZOS for good quality-speed balance
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Calculate padding once
        pad_width = self.target_w - new_width
        pad_height = self.target_h - new_height
        pad_left = pad_width >> 1  # Bit shift for faster division
        pad_top = pad_height >> 1
        
        # Create new image with padding - faster than F.pad for this use case
        new_img = Image.new('RGB', self.size, (114, 114, 114))
        new_img.paste(resized_img, (pad_left, pad_top))
        
        return new_img, scale, (pad_left, pad_top)

class CVManager:
    def __init__(self, batch_size: int = 2):
        # SAHI-compatible detection model
        yolo_model_path = "best_v7.pt"
        sahi_model_type = 'ultralytics'  
        
        # Load the YOLO model
        self.yolo_model = YOLO(yolo_model_path)
        self.preprocessor = FastImagePreprocessor()
        
        # Set batch size limit
        self.batch_size = batch_size
        
        # Batch collection for accumulating images
        self.batch_buffer = []
        self.batch_results = []
        
        confidence_threshold = 0.15
        # Pre-compile transforms for speed
        self.tensor_transform = T.Compose([
            T.ToTensor(),
        ])  # Skip normalization for speed unless absolutely necessary
        
        # Cache device to avoid repeated checks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # SAHI settings (commented out as mentioned)
        self.sahi_detection_model = None
        self.sahi_temp_dir = "temp_sahi_images"
        os.makedirs(self.sahi_temp_dir, exist_ok=True)
        

        try:
            self.sahi_detection_model = AutoDetectionModel.from_pretrained(
                model_type=sahi_model_type,
                model_path=yolo_model_path,
                confidence_threshold=confidence_threshold,
                device=device,
            )
            log.info("sahi works")
        except Exception as e:
            log.info(f"Failed to initialize SAHI model: {e}")
            self.sahi_detection_model = None
            
        # Directory for temporarily storing input images for SAHI
        self.sahi_temp_dir = "temp_sahi_images"
        os.makedirs(self.sahi_temp_dir, exist_ok=True)
        
    def cv(self, image: bytes, use_sahi: bool = True) -> list[dict[str, Any]]:
        """Performs object detection on an image with batch processing.
        
        Args:
            image: The image file in bytes.
            use_sahi: Whether to use SAHI for sliced inference (True) or regular YOLO (False).
            
        Returns:
            A list of predictions with required format:
            - bbox: [x, y, w, h] - Bounding box coordinates and dimensions in pixels
            - category_id: Index of the predicted category
            - score: Confidence score (only returned with SAHI)
        """
        # Add image to batch buffer
        self.batch_buffer.append(image)
        
        # If batch is full, process it
        if len(self.batch_buffer) >= self.batch_size:
            self._process_current_batch(use_sahi)
        
        # Return result for the current image (last added)
        if len(self.batch_results) >= len(self.batch_buffer):
            # Get the result for this specific image
            result_index = len(self.batch_buffer) - 1
            return self.batch_results[result_index]
        else:
            # If batch not yet processed, process single image immediately
            if use_sahi and self.sahi_detection_model:
                return self._process_with_sahi(image)
            else:
                return self._process_with_yolo(image)
    
    def flush_batch(self, use_sahi: bool = True) -> None:
        """Process any remaining images in the batch buffer."""
        if self.batch_buffer:
            self._process_current_batch(use_sahi)
    
    def _process_current_batch(self, use_sahi: bool = True) -> None:
        """Process the current batch buffer."""
        if not self.batch_buffer:
            return
            
        if use_sahi and self.sahi_detection_model:
            results = self._process_batch_with_sahi(self.batch_buffer)
        else:
            results = self._process_batch_with_yolo(self.batch_buffer)
        
        # Store results and clear buffer
        self.batch_results = results
        self.batch_buffer = []
    
    def _process_batch_with_yolo_chunked(self, images: list[bytes]) -> list[list[dict[str, Any]]]:
        """Process multiple images using batch YOLO inference with batch size limiting."""
        all_results = []
        
        # Process images in chunks of batch_size
        for i in range(0, len(images), self.batch_size):
            batch_chunk = images[i:i + self.batch_size]
            chunk_results = self._process_batch_with_yolo(batch_chunk)
            all_results.extend(chunk_results)
        
        return all_results
        """Process multiple images using batch YOLO inference."""
        batch_tensors = []
        batch_metadata = []
        
        # Prepare all images for batch processing
        for image_bytes in images:
            pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            original_width, original_height = pil_image.size
            
            # Apply preprocessing if enabled
            if preprocess:
                pil_image = self.preprocessor.preprocess_pipeline(pil_image, noise_level)
               
            # Apply letterboxing transform
            letterbox = LetterboxTransform(size=(1280, 1280))
            padded_img, scale, padding = letterbox(pil_image)
            pad_left, pad_top = padding
            
            # Convert to tensor
            tensor = self.tensor_transform(padded_img)
            batch_tensors.append(tensor)
            
            # Store metadata for coordinate conversion
            batch_metadata.append({
                'scale': scale,
                'padding': (pad_left, pad_top),
                'original_size': (original_width, original_height)
            })
        
        # Stack tensors into batch
        batch_tensor = torch.stack(batch_tensors).to(self.device)
        
        # Run batch prediction
        results = self.yolo_model.predict(
            batch_tensor, 
            conf=conf_threshold,
            iou=0.45,  # NMS IoU threshold
            max_det=300,  # Increase max detections for small objects
        )
        
        # Process results for each image
        batch_output = []
        for i, result in enumerate(results):
            metadata = batch_metadata[i]
            scale = metadata['scale']
            pad_left, pad_top = metadata['padding']
            
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                batch_output.append([])
                continue
                
            # Get coordinates (in xyxy format: x1, y1, x2, y2)
            bounding_boxes_xyxy = boxes.xyxy
            
            # Get confidence scores and classes
            confidence = boxes.conf
            classes = boxes.cls
            
            image_output = []
            
            # Process each detection for this image
            for pred_index in range(len(boxes)):
                # Extract coordinates from model output (these are in padded image space)
                x1, y1, x2, y2 = [float(coord) for coord in bounding_boxes_xyxy[pred_index]]
                
                # Adjust for padding
                x1 = (x1 - pad_left) if x1 > pad_left else 0
                y1 = (y1 - pad_top) if y1 > pad_top else 0
                x2 = (x2 - pad_left) if x2 > pad_left else 0
                y2 = (y2 - pad_top) if y2 > pad_top else 0
                
                # Convert back to original image coordinates
                x1 /= scale
                y1 /= scale
                x2 /= scale
                y2 /= scale
                
                # Calculate width and height
                w = x2 - x1
                h = y2 - y1
                
                # Get category ID and confidence
                category_id = int(classes[pred_index].item())
                score = float(confidence[pred_index].item())
                
                # Create prediction dictionary in required format
                pred_dict = {
                    "bbox": [x1, y1, w, h],
                    "category_id": category_id,
                    "score": score
                }
                
                image_output.append(pred_dict)
            
            batch_output.append(image_output)
            
        return batch_output
    
    def _process_batch_with_sahi(self, images: list[bytes]) -> list[list[dict[str, Any]]]:
        """Process multiple images using SAHI (sequential processing since SAHI doesn't support batching)."""
        if not self.sahi_detection_model:
            print("SAHI model not initialized. Falling back to batch YOLO.")
            return self._process_batch_with_yolo(images)
        
        batch_results = []
        for image_bytes in images:
            try:
                result = self._process_with_sahi(image_bytes)
                batch_results.append(result)
            except Exception as e:
                print(f"SAHI processing failed for image: {e}")
                print("Falling back to standard YOLO for this image.")
                result = self._process_with_yolo(image_bytes)
                batch_results.append(result)
        
        return batch_results
            
    def _process_with_yolo(self, image: bytes) -> list[dict[str, Any]]:
        """Process image using standard YOLO inference."""
        # Process the image
        pil_image = Image.open(io.BytesIO(image)).convert('RGB')
        original_width, original_height = pil_image.size
        
        # Apply preprocessing if enabled
        if preprocess:
            pil_image = self.preprocessor.preprocess_pipeline(pil_image, noise_level)
           
        # Apply letterboxing transform
        letterbox = LetterboxTransform(size=(1280, 1280))
        padded_img, scale, padding = letterbox(pil_image)
        pad_left, pad_top = padding
        
        # Convert to tensor
        tensor = self.tensor_transform(padded_img).unsqueeze(0).to(self.device)
        tensor = transform(padded_img).unsqueeze(0)  # Add batch dimension
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensor = tensor.to(device)
        
        # Run prediction
        results = self.yolo_model.predict(
            tensor, 
            conf=conf_threshold,
            iou=0.45,  # NMS IoU threshold
            max_det=300,  # Increase max detections for small objects
        )
        result = results[0]
        boxes = result.boxes
        
        # Get coordinates (in xyxy format: x1, y1, x2, y2)
        bounding_boxes_xyxy = boxes.xyxy
        
        # Get confidence scores and classes
        confidence = boxes.conf
        classes = boxes.cls
        
        output = []
        
        # Process each detection
        for pred_index in range(len(boxes)):
            # Extract coordinates from model output (these are in padded image space)
            x1, y1, x2, y2 = [float(coord) for coord in bounding_boxes_xyxy[pred_index]]
            
            # Adjust for padding
            x1 = (x1 - pad_left) if x1 > pad_left else 0
            y1 = (y1 - pad_top) if y1 > pad_top else 0
            x2 = (x2 - pad_left) if x2 > pad_left else 0
            y2 = (y2 - pad_top) if y2 > pad_top else 0
            
            # Convert back to original image coordinates
            x1 /= scale
            y1 /= scale
            x2 /= scale
            y2 /= scale
            
            # Calculate width and height
            w = x2 - x1
            h = y2 - y1
            
            # Get category ID and confidence
            category_id = int(classes[pred_index].item())
            score = float(confidence[pred_index].item())
            
            # Create prediction dictionary in required format
            pred_dict = {
                "bbox": [x1, y1, w, h],
                "category_id": category_id,
                "score": score
            }
            
            output.append(pred_dict)
            
        return output
    
    def _process_with_sahi(self, image: bytes) -> list[dict[str, Any]]:
        """Process image using SAHI sliced inference."""
        if not self.sahi_detection_model:
            print("SAHI model not initialized. Falling back to standard YOLO.")
            return self._process_with_yolo(image)
            
        try:
            # Preprocess image if enabled
            pil_image = Image.open(io.BytesIO(image)).convert('RGB')
            if preprocess:
                pil_image = self.preprocessor.preprocess_pipeline(pil_image, noise_level)
            
            # Save preprocessed image temporarily
            temp_image_path = os.path.join(self.sahi_temp_dir, f"{uuid.uuid4()}.jpg")
            pil_image.save(temp_image_path, quality=95)
                
            # Get predictions using SAHI sliced inference
            result = get_sliced_prediction(
                temp_image_path,
                self.sahi_detection_model,
                slice_height=640,  # Larger slices for better context
                slice_width=640,
                overlap_height_ratio=0.3,  # Increased overlap
                overlap_width_ratio=0.3,
                postprocess_type="NMS",  # Use NMS for post-processing
                postprocess_match_metric="IOS",  # Intersection over smaller area
                postprocess_match_threshold=0.5,
                postprocess_class_agnostic=False,
            )
            
            # Convert to COCO predictions format
            coco_predictions = result.to_coco_predictions(image_id=0)
            
            # Format the results according to COCO format
            detections = []
            for pred in coco_predictions:
                # COCO format bbox is [x, y, width, height]
                detections.append({
                    "bbox": pred["bbox"],  # Already in [x, y, w, h] format
                    "category_id": pred["category_id"],
                    "score": pred["score"]
                })
            
            # Clean up temporary file
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
                
            return detections
            
        except Exception as e:
            print(f"SAHI processing failed: {e}")
            print("Falling back to standard YOLO.")
            return self._process_with_yolo(image)

    def batch_inference(self, images: list[bytes], preprocessing: str = 'minimal') -> list[list[dict]]:
        """Process multiple images in batch for better speed."""
        results = []
        
        # Process images without recreating objects
        for image_bytes in images:
            detections = self._process_with_yolo_fast(image_bytes, preprocessing)
            results.append(detections)
            
        return results
