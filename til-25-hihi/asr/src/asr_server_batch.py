"""Runs the ASR server."""

# Unless you want to do something special with the server, you shouldn't need
# to change anything in this file.


from fastapi import FastAPI, Request
from asr_manager_batch import ASRManager


app = FastAPI()
manager = ASRManager()


@app.post("/asr")
async def asr(request: Request) -> dict[str, list[str]]:
    """Performs ASR on audio files.

    Args:
        request: The API request. Contains a list of audio files, encoded in
            base-64.

    Returns:
        A `dict` with a single key, `"predictions"`, mapping to a `list` of
        `str` transcriptions, in the same order as which appears in `request`.
    """

    inputs_json = await request.json()
 
    trans = manager.asr(inputs_json["instances"])

    return {"predictions": trans}

@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint for the server."""
    return {"message": "health ok"}
