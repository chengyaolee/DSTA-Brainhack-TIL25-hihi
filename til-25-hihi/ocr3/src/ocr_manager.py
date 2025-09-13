# src/ocr_manager.py - Enhanced version with better layout handling
import cv2
import numpy as np
import pytesseract
from PIL import Image

class OCRManager:
    def __init__(self):
        print("INFO: OCRManager initialized.")

    def preprocess_image(self, img):
        """Preprocess image for better OCR results"""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise
        denoised = cv2.medianBlur(thresh, 1)
        
        return denoised

    def detect_text_blocks(self, img):
        """Detect text blocks using morphological operations"""
        # Preprocess
        processed = self.preprocess_image(img)
        
        # Create rectangular kernel for morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))
        
        # Apply dilation to connect text regions
        dilated = cv2.dilate(processed, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter and sort contours by area and position
        valid_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area > 500:  # Filter small regions
                valid_contours.append((x, y, w, h))
        
        # Sort by y-coordinate (top to bottom), then x-coordinate (left to right)
        valid_contours.sort(key=lambda c: (c[1], c[0]))
        
        return valid_contours

    def ocr(self, image_bytes: bytes) -> str:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_np is None:
            print("ERROR: Could not decode image from bytes in OCR method.")
            return "ERROR_DECODING_IMAGE"

        try:
            # Detect text blocks
            text_blocks = self.detect_text_blocks(img_np)
            
            extracted_texts = []
            
            if text_blocks:
                # Process each detected block
                for x, y, w, h in text_blocks:
                    # Add padding to avoid cutting off text
                    padding = 5
                    y1 = max(0, y - padding)
                    y2 = min(img_np.shape[0], y + h + padding)
                    x1 = max(0, x - padding)
                    x2 = min(img_np.shape[1], x + w + padding)
                    
                    # Crop the region
                    cropped = img_np[y1:y2, x1:x2]
                    
                    # Convert to PIL
                    pil_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                    
                    # OCR with PSM 6 (uniform block)
                    text = pytesseract.image_to_string(pil_img, config='--psm 6').strip()
                    
                    if text:
                        extracted_texts.append(text)
            else:
                # Fallback to full image OCR
                pil_img = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
                text = pytesseract.image_to_string(pil_img, config='--psm 3')
                extracted_texts.append(text.strip())
            
            return '\n'.join(extracted_texts)
            
        except Exception as e:
            print(f"ERROR: Exception during OCR: {e}")
            return f"ERROR_PREDICT_FAILED: {e}"