import base64
from fastapi import FastAPI, Request
from ocr_manager import OCRManager

app = FastAPI()

# Initialize OCR manager with error handling
try:
    manager = OCRManager()
except Exception:
    manager = None

@app.post("/ocr")
async def ocr(request: Request) -> dict[str, list[str]]:
    """OCR endpoint with simple error handling"""
    
    # Check if OCR manager is available
    if manager is None:
        return {"predictions": []}
    
    try:
        # Parse request
        inputs_json = await request.json()
        
        # Validate basic structure
        if "instances" not in inputs_json or not isinstance(inputs_json["instances"], list):
            return {"predictions": []}
        
        predictions = []
        
        # Process each instance
        for instance in inputs_json["instances"]:
            try:
                # Validate instance
                if not isinstance(instance, dict) or "b64" not in instance:
                    predictions.append("")
                    continue
                
                # Decode image
                image_bytes = base64.b64decode(instance["b64"])
                
                # Perform OCR
                text = manager.ocr(image_bytes)
                predictions.append(text if text is not None else "")
                
            except Exception:
                # If any instance fails, just add empty string
                predictions.append("")
        
        return {"predictions": predictions}
        
    except Exception:
        # If anything fails, return empty predictions
        return {"predictions": []}

@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint"""
    try:
        status = "ok" if manager is not None else "degraded"
        return {"message": status}
    except Exception:
        return {"message": "error"}