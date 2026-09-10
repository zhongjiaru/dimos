#!/usr/bin/env python3
# Copyright 2025-2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
from collections.abc import Iterable, Iterator
from importlib import import_module
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import numpy as np

DEFAULT_MODEL = "ASLP-lab/Cosyvoice2-Yue-ZoengJyutGaai"
_TRAINING_METADATA_KEYS = frozenset({"epoch", "step"})
_MODEL_WEIGHT_FILES = frozenset({"llm.pt", "flow.pt", "hift.pt"})


def _resolve_model(model_dir: str) -> str:
    local_model = Path(model_dir).expanduser()
    if local_model.is_dir():
        return str(local_model.resolve())
    hub = import_module("huggingface_hub")
    return str(hub.snapshot_download(repo_id=model_dir))


def _load_cosyvoice2(cosyvoice_module: Any, model_dir: str, **kwargs: Any) -> Any:
    """Load a model while tolerating metadata in fine-tuned LLM checkpoints."""
    torch: Any = import_module("torch")
    original_load = torch.load

    def compatible_load(path: Any, *args: Any, **load_kwargs: Any) -> Any:
        filename = ""
        if isinstance(path, (str, bytes, os.PathLike)):
            filename = Path(os.fsdecode(path)).name
        if filename in _MODEL_WEIGHT_FILES:
            # CosyVoice normally deserializes each checkpoint directly to CUDA.
            # Loading on CPU first avoids holding both the checkpoint and module
            # parameters in GPU memory during startup.
            load_kwargs["map_location"] = "cpu"
        checkpoint = original_load(path, *args, **load_kwargs)
        if filename == "llm.pt":
            if isinstance(checkpoint, dict):
                metadata = _TRAINING_METADATA_KEYS.intersection(checkpoint)
                if metadata:
                    logging.info(
                        "Ignoring CosyVoice training checkpoint metadata: %s",
                        ", ".join(sorted(metadata)),
                    )
                    return {
                        key: value
                        for key, value in checkpoint.items()
                        if key not in _TRAINING_METADATA_KEYS
                    }
        return checkpoint

    torch.load = compatible_load
    try:
        return cosyvoice_module.CosyVoice2(model_dir, **kwargs)
    finally:
        torch.load = original_load


def _audio_bytes(
    model_output: Iterable[dict[str, Any]],
    *,
    cleanup_path: Path | None = None,
) -> Iterator[bytes]:
    try:
        for output in model_output:
            speech = output["tts_speech"].detach().float().cpu().numpy()
            yield (np.clip(speech, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    finally:
        if cleanup_path is not None:
            cleanup_path.unlink(missing_ok=True)


def _build_app(model: Any) -> Any:
    fastapi = import_module("fastapi")
    responses = import_module("fastapi.responses")
    app = fastapi.FastAPI()

    @app.get("/health")  # type: ignore[untyped-decorator]
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "sample_rate": model.sample_rate,
            "speakers": model.list_available_spks(),
        }

    @app.post("/inference_sft")  # type: ignore[untyped-decorator]
    async def inference_sft(
        tts_text: str = fastapi.Form(),
        spk_id: str = fastapi.Form(),
        stream: bool = fastapi.Form(True),
        speed: float = fastapi.Form(1.0),
        text_frontend: bool = fastapi.Form(True),
    ) -> Any:
        output = model.inference_sft(
            tts_text,
            spk_id,
            stream=stream,
            speed=speed,
            text_frontend=text_frontend,
        )
        return responses.StreamingResponse(_audio_bytes(output))

    @app.post("/inference_instruct2")  # type: ignore[untyped-decorator]
    async def inference_instruct2(
        tts_text: str = fastapi.Form(),
        instruct_text: str = fastapi.Form(),
        prompt_wav: Any = fastapi.File(),
        stream: bool = fastapi.Form(True),
        speed: float = fastapi.Form(1.0),
        text_frontend: bool = fastapi.Form(True),
    ) -> Any:
        with tempfile.NamedTemporaryFile(
            prefix="dimos-cosyvoice2-prompt-",
            suffix=".wav",
            delete=False,
        ) as prompt_file:
            shutil.copyfileobj(prompt_wav.file, prompt_file)
            prompt_path = Path(prompt_file.name)
        try:
            output = model.inference_instruct2(
                tts_text,
                instruct_text,
                str(prompt_path),
                stream=stream,
                speed=speed,
                text_frontend=text_frontend,
            )
        except Exception:
            prompt_path.unlink(missing_ok=True)
            raise
        return responses.StreamingResponse(
            _audio_bytes(output, cleanup_path=prompt_path),
            media_type="application/octet-stream",
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve CosyVoice2-Yue for DimOS")
    parser.add_argument("--cosyvoice-repo", type=Path, required=True)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50000)
    parser.add_argument("--load-jit", action="store_true")
    parser.add_argument("--load-trt", action="store_true")
    parser.add_argument("--load-vllm", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--trt-concurrent", type=int, default=1)
    args = parser.parse_args()

    repo = args.cosyvoice_repo.expanduser().resolve()
    for source_dir in (repo, repo / "third_party" / "Matcha-TTS"):
        sys.path.insert(0, str(source_dir))

    cosyvoice_module = import_module("cosyvoice.cli.cosyvoice")
    model = _load_cosyvoice2(
        cosyvoice_module,
        _resolve_model(args.model_dir),
        load_jit=args.load_jit,
        load_trt=args.load_trt,
        load_vllm=args.load_vllm,
        fp16=args.fp16,
        trt_concurrent=args.trt_concurrent,
    )
    app = _build_app(model)
    uvicorn = import_module("uvicorn")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
