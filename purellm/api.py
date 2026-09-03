import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from purellm.torchgpt.checkpoint import load_model_checkpoint
from purellm.torchgpt.generation import generate

MAX_PROMPT_CHARACTERS = 8_192
MAX_NEW_TOKENS = 512


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(
        min_length=1,
        max_length=MAX_PROMPT_CHARACTERS,
        description="Text used to start generation.",
    )
    max_new_tokens: int = Field(
        default=64,
        ge=1,
        le=MAX_NEW_TOKENS,
        description="Maximum number of tokens to generate.",
    )
    temperature: float = Field(
        default=0.8,
        gt=0,
        le=2.0,
        description="Sampling temperature.",
    )

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("prompt must contain non-whitespace characters")
        return prompt


class GenerateResponse(BaseModel):
    text: str = Field(description="Prompt followed by its generated continuation.")
    generated_tokens: int
    latency_ms: float
    tokens_per_second: float


class HealthResponse(BaseModel):
    status: Literal["ok"]
    device: str
    checkpoint_step: int


def create_app(
    checkpoint_path: Path | None = None,
    *,
    device: torch.device | None = None,
) -> FastAPI:
    # ponytail: one lock matches the model's single KV cache; shard by replica if needed.
    generation_lock = Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        path = checkpoint_path or _checkpoint_path_from_environment()
        selected_device = device or _select_device()
        app.state.checkpoint = load_model_checkpoint(path, device=selected_device)
        app.state.device = selected_device
        yield
        app.state.checkpoint.model.reset_cache()

    app = FastAPI(
        title="PureLLM Inference API",
        description="Generate text with a single loaded PureLLM checkpoint.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get(
        "/health",
        response_model=HealthResponse,
        summary="Check inference readiness",
        tags=["service"],
    )
    def health(request: Request) -> HealthResponse:
        checkpoint = request.app.state.checkpoint
        return HealthResponse(
            status="ok",
            device=str(request.app.state.device),
            checkpoint_step=checkpoint.step,
        )

    @app.post(
        "/generate",
        response_model=GenerateResponse,
        summary="Generate a text continuation",
        tags=["inference"],
    )
    def generate_text(payload: GenerateRequest, request: Request) -> GenerateResponse:
        checkpoint = request.app.state.checkpoint
        device = request.app.state.device

        with generation_lock:
            started_at = perf_counter()
            try:
                prompt_ids = checkpoint.tokenizer.encode(payload.prompt)
            except KeyError as error:
                raise HTTPException(
                    status_code=422,
                    detail=f"prompt contains an unsupported character: {error.args[0]!r}",
                ) from error

            generated_ids = generate(
                model=checkpoint.model,
                prompt_ids=prompt_ids,
                max_new_tokens=payload.max_new_tokens,
                temperature=payload.temperature,
                device=device,
            )
            elapsed = perf_counter() - started_at

        generated_tokens = len(generated_ids) - len(prompt_ids)
        return GenerateResponse(
            text=checkpoint.tokenizer.decode(generated_ids, errors="replace"),
            generated_tokens=generated_tokens,
            latency_ms=elapsed * 1_000,
            tokens_per_second=generated_tokens / max(elapsed, 1e-9),
        )

    return app


def _checkpoint_path_from_environment() -> Path:
    checkpoint_path = os.environ.get("PURELLM_CHECKPOINT")
    if not checkpoint_path:
        raise RuntimeError("PURELLM_CHECKPOINT must point to a model checkpoint")
    return Path(checkpoint_path)


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


app = create_app()
