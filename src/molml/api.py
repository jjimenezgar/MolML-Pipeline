"""Small local HTTP API around the V3 inference function."""

from __future__ import annotations

from contextlib import asynccontextmanager
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from molml.artifacts import load_model_bundle
from molml.predict import predict_smiles


MAX_BATCH_SIZE = 128


class PredictRequest(BaseModel):
    smiles: list[str]


def create_app(*, model_path: str | Path | None = None, bundle: dict | None = None) -> FastAPI:
    """Build the API app; model artifacts are loaded once during startup.

    ``bundle`` is intended for tests or embedding. Normal use sets
    ``MOLML_MODEL_PATH`` to a trusted V3 joblib bundle.
    """
    if bundle is not None and model_path is not None:
        raise ValueError("Pass either bundle or model_path, not both.")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if bundle is not None:
            app.state.model_bundle = bundle
        else:
            selected_path = model_path or os.environ.get("MOLML_MODEL_PATH")
            app.state.model_bundle = load_model_bundle(selected_path) if selected_path else None
        yield

    app = FastAPI(
        title="MolML-Pipeline API",
        description="Local inference for the MoleculeNet BACE baseline. Scores are unvalidated model outputs.",
        version="1.5.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health(request: Request):
        model_bundle = request.app.state.model_bundle
        if model_bundle is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Set MOLML_MODEL_PATH to a trusted V3 model bundle.",
            )
        return {"status": "ok", "model_ready": True, "model": model_bundle["model_name"]}

    @app.post("/predict")
    def predict(payload: PredictRequest, request: Request):
        if not payload.smiles:
            raise HTTPException(status_code=422, detail="Provide at least one SMILES string.")
        if len(payload.smiles) > MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=422,
                detail=f"Batch limit is {MAX_BATCH_SIZE} SMILES per request.",
            )
        model_bundle = request.app.state.model_bundle
        if model_bundle is None:
            raise HTTPException(status_code=503, detail="Model is not loaded.")
        return {"predictions": predict_smiles(model_bundle, payload.smiles)}

    return app


app = create_app()


def main():
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("MOLML_API_HOST", "127.0.0.1"),
        port=int(os.environ.get("MOLML_API_PORT", "8000")),
    )


if __name__ == "__main__":
    main()
