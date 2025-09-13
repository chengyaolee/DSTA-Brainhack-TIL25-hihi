# src/ocr_server.py
import base64
from fastapi import FastAPI, Request, HTTPException
from .ocr_manager import OCRManager
import logging

app = FastAPI(
    title="OCR Service",
    version="1.0.0",
    description="Performs OCR on images using Tesseract."
)

manager = OCRManager()
logger = logging.getLogger("uvicorn.error")

@app.post("/ocr", summary="Perform OCR on an image")
async def ocr_endpoint(request: Request) -> dict[str, list[str]]:
    """
    Receives a JSON payload with base64 encoded image(s) and returns OCR predictions.
    - Input: `{"instances": [{"b64": "<base64_image_string>"}]}`
    - Output: `{"predictions": ["<extracted_text_string>"]}`
    """
    try:
        inputs_json = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")
        
    predictions = []
    
    for instance in inputs_json.get("instances", []):
        image_b64 = instance.get("b64")
        if image_b64 and isinstance(image_b64, str):
            try:
                image_bytes = base64.b64decode(image_b64)
                text = manager.ocr(image_bytes)
                predictions.append(text)
            except base64.binascii.Error:
                logger.warning("Invalid base64 string received for an instance.")
                predictions.append("")
            except Exception as e:
                logger.error(f"OCR processing error for an instance: {e}")
                predictions.append("")
        else:
            if "b64" in instance:
                logger.warning("Missing or invalid 'b64' field in an instance.")
            predictions.append("")

    return {"predictions": predictions}

@app.get("/health", summary="Health check endpoint")
def health_check() -> dict[str, str]:
    """Returns a simple health status, indicating the service is operational."""
    return {"status": "ok", "message": "OCR Service is running"}