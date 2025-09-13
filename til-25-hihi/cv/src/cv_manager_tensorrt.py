"""Manages the CV model with TensorRT support."""
from typing import Any
import os
import io
from PIL import Image
import numpy as np
import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # Automatically initializes CUDA context
import torchvision.transforms as T
import torchvision.transforms.functional as F
import uuid
import logging as log

class LetterboxTransform:
    """
    Letterbox transform that resizes and pads the image to maintain aspect ratio.
    Similar to YOLO's letterboxing approach.
    """
    def __init__(self, size=(1280, 1280)):
        self.size = size
        
    def __call__(self, img):
        # Calculate the padding needed to maintain aspect ratio
        width, height = img.size
        target_w, target_h = self.size
        
        # Calculate scaling factor to maintain aspect ratio
        scale = min(target_w / width, target_h / height)
        
        # Calculate new size
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize image maintaining aspect ratio
        resized_img = F.resize(img, [new_height, new_width])
        
        # Calculate padding
        pad_width = target_w - new_width
        pad_height = target_h - new_height
        
        # Padding on each side
        pad_left = pad_width // 2
        pad_top = pad_height // 2
        pad_right = pad_width - pad_left
        pad_bottom = pad_height - pad_top
        
        # Apply padding
        padded_img = F.pad(resized_img, [pad_left, pad_top, pad_right, pad_bottom], 0)
        
        return padded_img, scale, (pad_left, pad_top)

class CVManager:
    def __init__(self, engine_path: str = "model.engine"):
        """Initialize with a TensorRT engine file."""
        # Load TensorRT engine
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        self.input_shape = self.engine.get_binding_shape(0)  # Assuming one input
        self.output_shape = self.engine.get_binding_shape(1)  # Assuming one output
        
        # Allocate device memory for inputs and outputs
        self.input_size = trt.volume(self.input_shape) * np.dtype(np.float32).itemsize
        self.output_size = trt.volume(self.output_shape) * np.dtype(np.float32).itemsize
        self.d_input = cuda.mem_alloc(self.input_size)
        self.d_output = cuda.mem_alloc(self.output_size)
        
        # Temporary directory (kept for compatibility, though SAHI is disabled here)
        self.sahi_temp_dir = "temp_sahi_images"
        os.makedirs(self.sahi_temp_dir, exist_ok=True)
        
        # SAHI is not supported with TensorRT in this implementation
        self.sahi_detection_model = None
        log.info("Initialized CVManager with TensorRT engine.")

    def cv(self, image: bytes, use_sahi: bool = False) -> list[dict[str, Any]]:
        """Performs object detection on an image using TensorRT.
        
        Args:
            image: The image file in bytes.
            use_sahi: Ignored in this TensorRT implementation (kept for compatibility).
            
        Returns:
            A list of predictions with required format:
            - bbox: [x, y, w, h] - Bounding box coordinates and dimensions in pixels
            - category_id: Index of the predicted category
            - score: Confidence score
        """
        if use_sahi:
            log.warning("SAHI is not supported with TensorRT. Using standard inference.")
        return self._process_with_tensorrt(image)

    def _process_with_tensorrt(self, image: bytes) -> list[dict[str, Any]]:
        """Process image using TensorRT inference."""
        # Process the image
        pil_image = Image.open(io.BytesIO(image)).convert('RGB')
        original_width, original_height = pil_image.size
        
        # Apply letterboxing transform (assuming model expects 1280x1280)
        letterbox = LetterboxTransform(size=(1280, 1280))
        padded_img, scale, padding = letterbox(pil_image)
        pad_left, pad_top = padding
        
        # Convert to numpy array and preprocess for TensorRT
        transform = T.Compose([T.ToTensor()])
        tensor = transform(padded_img).numpy()  # Shape: [3, 1280, 1280]
        input_data = np.transpose(tensor, (1, 2, 0)).astype(np.float32)  # HWC format
        input_data = np.ascontiguousarray(input_data)  # Ensure contiguous memory
        
        # Copy input data to device
        cuda.memcpy_htod(self.d_input, input_data)
        
        # Execute inference
        self.context.execute_v2([int(self.d_input), int(self.d_output)])
        
        # Copy output back to host
        output = np.empty(self.output_shape, dtype=np.float32)
        cuda.memcpy_dtoh(output, self.d_output)
        
        # Post-process output (assuming YOLO-like output: [num_detections, (x, y, w, h, conf, cls)])
        detections = self._postprocess_tensorrt_output(output, scale, pad_left, pad_top, original_width, original_height)
        
        return detections
    
    def _postprocess_tensorrt_output(self, output, scale, pad_left, pad_top, original_width, original_height) -> list[dict[str, Any]]:
        """Convert TensorRT output to the required format."""
        # Example post-processing (adjust based on your model's output format)
        # Assuming output is [num_detections, 6] with [x_center, y_center, w, h, conf, cls]
        output = output.reshape(-1, 6)  # Flatten to [num_detections, 6]
        
        detections = []
        for det in output:
            x_center, y_center, w, h, conf, cls = det
            if conf < 0.15:  # Confidence threshold
                continue
            
            # Adjust coordinates from padded image space to original image space
            x_center = (x_center - pad_left) / scale
            y_center = (y_center - pad_top) / scale
            w /= scale
            h /= scale
            
            # Convert center coordinates to top-left corner
            x1 = x_center - w / 2
            y1 = y_center - h / 2
            
            # Ensure coordinates are within bounds
            x1 = max(0, min(x1, original_width))
            y1 = max(0, min(y1, original_height))
            w = min(w, original_width - x1)
            h = min(h, original_height - y1)
            
            detections.append({
                "bbox": [x1, y1, w, h],
                "category_id": int(cls),
                "score": float(conf)
            })
        
        return detections