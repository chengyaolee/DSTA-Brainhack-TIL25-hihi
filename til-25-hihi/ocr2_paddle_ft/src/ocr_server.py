import base64
from fastapi import FastAPI, Request
from ocr_manager import OCRManager

app = FastAPI()
manager = OCRManager()


@app.post("/ocr")
async def ocr(request: Request) -> dict[str, list[str]]:
    inputs_json = await request.json()
    predictions = []
    
    for instance in inputs_json["instances"]:
        image_bytes = base64.b64decode(instance["b64"])
        text = manager.ocr(image_bytes)
        predictions.append(text)

    return {"predictions": predictions}


@app.get("/health")
def health() -> dict[str, str]:
    return {"message": "ok"}
