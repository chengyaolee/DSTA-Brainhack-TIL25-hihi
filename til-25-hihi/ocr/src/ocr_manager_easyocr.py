# src/ocr_manager.py

import os
import cv2
import numpy as np
import torch
import easyocr
import base64

class OCRManager:
    def __init__(self):
        print("Initializing OCRManager...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.ocr_engine= easyocr.Reader(
            lang_list=['en'],
            model_storage_directory='EasyOCR',
            download_enabled=False,
            user_network_directory='EasyOCR',
            recog_network='english_g2'
        )
        
        # # Load custom detection and recognition model
        # det_model = db_resnet50(pretrained=False, pretrained_backbone=False)
        # # det_params = torch.load('<path_to_pt>', map_location=device)
        # # det_model.load_state_dict(det_params)
        # reco_model = crnn_vgg16_bn(pretrained=False, pretrained_backbone=False)
        # # reco_params = torch.load('<path_to_pt>', map_location=device)
        # # reco_model.load_state_dict(reco_params)

        print("EasyOCR engine initialized successfully in OCRManager.")

    def predict_batch(self, instances: list) -> dict:
        """
        Processes a batch of images provided as base64 encoded strings.

        Args:
            instances (list): A list of dictionaries, where each dictionary has a 'b64' key 
                              containing a base64 encoded JPEG image string. torch
                              Example: [{"key": 0, "b64": "BASE64_IMAGE_DATA"}, ...]

        Returns:
            dict: A dictionary with a "predictions" key, containing a list of
                  transcribed text strings, in the same order as the input instances.
                  Example: {"predictions": ["text1", "text2", ...]}
        """
        predictions = []
        for i, instance_data in enumerate(instances):
            b64_string = instance_data.get("b64")
            # instance_key = instance_data.get("key", f"item_{i}") # Key for logging if needed

            if not b64_string:
                print(f"Warning: Missing 'b64' data for instance {i}. Skipping.")
                predictions.append("") # Append empty string for missing data to maintain order
                continue
       
            try:
                # Decode base64 image
                try:
                    image_bytes = base64.b64decode(b64_string)
                    # Convert bytes to OpenCV image
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img_cv is None:
                        raise ValueError("cv2.imdecode returned None. Invalid image data or format.")
                except Exception as e:
                    print(f"Error decoding base64 string or loading image for instance {i}: {e}")
                    predictions.append("") 
                    continue
                print("testing")
                image_height, image_width, _ = img_cv.shape
                print(f"OCRManager: Decoded image dimensions: {image_width}x{image_height}")

                results = self.ocr_engine.readtext(img_np) # Pass NumPy array            
                predictions.append(results)

            except RuntimeError as e: # Catch errors from _generate_hocr_file or other critical steps
                print(f"RuntimeError processing instance {i}: {e}")
                predictions.append("") # Append empty string on error
            except Exception as e:
                print(f"Unexpected error processing instance {i}: {e}")
                predictions.append("")
        return {"predictions": predictions}
