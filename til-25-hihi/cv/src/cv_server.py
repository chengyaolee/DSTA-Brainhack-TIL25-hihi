"""Runs the CV server."""

import base64
import os
import sys
from typing import Any

from fastapi import FastAPI, Request

from cv_manager import CVManager

# Adjust sys.path to include Real-ESRGAN directory
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Real-ESRGAN')))
# import inference_realesrgan as Upscale

app = FastAPI()
manager = CVManager()

@app.post("/cv")
async def cv(request: Request) -> dict[str, list[list[dict[str, Any]]]]:
    """Performs CV object detection on image frames.

    Args:
        request: The API request. Contains a list of images, encoded in
            base-64.

    Returns:
        A `dict` with a single key, `"predictions"`, mapping to a `list` of
        `dict`s containing your CV model's predictions, in the same order as
        which appears in `request`. See `cv/README.md` for the expected format.
    """
    inputs_json = await request.json()
    predictions = []

    for instance in inputs_json["instances"]:
        # Decode base64 image to bytes
        image_bytes = base64.b64decode(instance["b64"])

        # Upscale image using Real-ESRGAN
        # try:
        #     upscaled_image_bytes = Upscale.upscale_image_bytes(
        #         image_bytes,
        #         model_name="realesr-general-x4v3",
        #         outscale=4
        #     )
        # except Exception as e:
        #     print(f"Upscaling failed: {e}")
        #     continue  # Skip to next image on failure

        # Perform object detection on the upscaled image
        detections = manager.cv(image_bytes)
        predictions.append(detections)

    return {"predictions": predictions}

@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint for your model."""
    return {"message": "health ok"}