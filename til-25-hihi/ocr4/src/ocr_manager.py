# src/ocr_manager.py - Enhanced version with better layout analysis
import cv2
import numpy as np
import easyocr
import torch
from typing import List, Tuple

class OCRManager:
    def __init__(self):
        # Check if GPU is available
        self.use_gpu = torch.cuda.is_available()
        print(f"INFO: Using GPU: {self.use_gpu}")
        
        # Initialize EasyOCR reader with optimal settings
        self.reader = easyocr.Reader(
            ['en'],  # Languages
            gpu=self.use_gpu,
            verbose=False,
            quantize=False  # Better accuracy
        )
        print("INFO: OCRManager initialized with EasyOCR.")

    def group_text_by_blocks(self, results: List[Tuple]) -> List[str]:
        """Group OCR results into logical blocks based on spatial proximity"""
        if not results:
            return []
        
        # Extract bounding boxes and texts
        blocks = []
        for bbox, text, conf in results:
            # Get the top-left and bottom-right coordinates
            x_min = min(point[0] for point in bbox)
            y_min = min(point[1] for point in bbox)
            x_max = max(point[0] for point in bbox)
            y_max = max(point[1] for point in bbox)
            
            blocks.append({
                'bbox': (x_min, y_min, x_max, y_max),
                'text': text.strip(),
                'confidence': conf,
                'merged': False
            })
        
        # Sort by y-coordinate first, then x-coordinate
        blocks.sort(key=lambda b: (b['bbox'][1], b['bbox'][0]))
        
        # Merge nearby blocks
        merged_blocks = []
        for block in blocks:
            if block['merged'] or not block['text']:
                continue
                
            current_group = [block['text']]
            block['merged'] = True
            
            # Look for nearby blocks to merge
            for other in blocks:
                if other['merged'] or not other['text']:
                    continue
                    
                # Check if blocks are on similar horizontal line
                y_overlap = (
                    block['bbox'][1] <= other['bbox'][3] and 
                    block['bbox'][3] >= other['bbox'][1]
                )
                
                # Check horizontal distance
                x_distance = min(
                    abs(block['bbox'][2] - other['bbox'][0]),
                    abs(other['bbox'][2] - block['bbox'][0])
                )
                
                # Merge if on same line and close horizontally
                if y_overlap and x_distance < 50:
                    current_group.append(other['text'])
                    other['merged'] = True
            
            # Join the group texts
            merged_text = ' '.join(current_group)
            if merged_text:
                merged_blocks.append(merged_text)
        
        return merged_blocks

    def ocr(self, image_bytes: bytes) -> str:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_np is None:
            print("ERROR: Could not decode image from bytes in OCR method.")
            return "ERROR_DECODING_IMAGE"

        try:
            # Perform OCR with detailed results
            results = self.reader.readtext(
                img_np,
                paragraph=False,  # Get individual text elements
                detail=1,         # Return full details
                text_threshold=0.7,
                low_text=0.4,
                link_threshold=0.4,
                canvas_size=2560,
                mag_ratio=1.5
            )
            
            # Group text into logical blocks
            text_blocks = self.group_text_by_blocks(results)
            
            return "\n".join(text_blocks)
            
        except Exception as e:
            print(f"ERROR: Exception during OCR: {e}")
            return f"ERROR_PREDICT_FAILED: {e}"