# src/ocr_manager.py

import os
import cv2
import numpy as np
from paddleocr import PaddleOCR

class OCRManager:
    def __init__(self):
        models_base_dir = os.getenv('MODELS_BASE_DIR')
        det_model_dir_path = os.path.join(models_base_dir, os.getenv('DET_MODEL_SUBDIR'))
        rec_model_dir_path = os.path.join(models_base_dir, os.getenv('REC_MODEL_SUBDIR'))
        cls_model_dir_path = os.path.join(models_base_dir, os.getenv('CLS_MODEL_SUBDIR'))
        layout_model_dir_path = os.path.join(models_base_dir, os.getenv('LAYOUT_MODEL_SUBDIR'))
        
        # Quick validation - if models don't exist, PaddleOCR will download them
        for path in [det_model_dir_path, cls_model_dir_path, layout_model_dir_path]:
            if not os.path.exists(os.path.join(path, 'inference.pdmodel')):
                print(f"Model not found at {path}")
        
        self.ocr_engine = PaddleOCR(
            use_angle_cls=False,
            lang='en',
            layout=True,
            show_log=False,
            use_gpu=True,
            det_model_dir=det_model_dir_path,
            rec_model_dir=rec_model_dir_path,
            rec_char_dict_path="/opt/paddleocr_models/dicts/custom_char_dict.txt",
            cls_model_dir=cls_model_dir_path,
            layout_model_dir=layout_model_dir_path
        )

    def ocr(self, image_bytes: bytes) -> str:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        ocr_result_batch = self.ocr_engine.ocr(img_np, cls=False)
        
        if not ocr_result_batch[0]:
            return ""
        
        extracted_texts = []
        for line_info in ocr_result_batch[0]:
            extracted_texts.append(line_info[1][0])

        return "\n".join(extracted_texts)