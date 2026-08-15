"""
FastAPI application for insurance LLM serving.

Endpoints:
- POST /v1/chat/completions — Chat with the model
- GET  /health              — Health check
- GET  /                    — API info
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import ChatRequest, ChatResponse, HealthResponse
from src.serving.inference import engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    logger.info("Starting model loading...")

    base_model = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
    adapter_repo = os.getenv("ADAPTER_REPO", "sefabilicier/insurance-qwen3b-dpo")
    hf_token = os.getenv("HF_TOKEN")
    local_adapter = os.getenv("LOCAL_ADAPTER_PATH")

    try:
        engine.load(
            base_model_name=base_model,
            adapter_repo=adapter_repo,
            hf_token=hf_token,
            local_adapter_path=local_adapter,
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")

    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Insurance LLM API",
    description="Fine-tuned Qwen2.5-3B for insurance customer support",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Info"])
async def root():
    return {
        "service": "Insurance LLM API",
        "model": engine.model_id or "not loaded",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    return HealthResponse(
        status="healthy" if engine.is_loaded else "model_not_loaded",
        model=engine.model_id or "none",
        device=engine.device,
        model_loaded=engine.is_loaded,
    )


@app.post("/v1/chat/completions", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """Chat with the insurance support model."""
    if not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    try:
        result = engine.generate(
            messages=messages,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )

        return ChatResponse(
            response=result["response"],
            model=engine.model_id or "unknown",
            usage=result["usage"],
        )
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))