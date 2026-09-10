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

from pathlib import Path
from typing import Self

from fastapi.testclient import TestClient
import numpy as np
import pytest

from dimos.stream.audio.tts.node_cosyvoice2_yue import (
    COSYVOICE2_YUE_MODEL,
    CosyVoice2YueHTTPNode,
    CosyVoice2YueTTSNode,
    _resolve_model_dir,
)
from scripts.serve_cosyvoice2_yue import _build_app, _load_cosyvoice2


class _FakeSpeech:
    def __init__(self, samples: list[float]) -> None:
        self._samples = np.array(samples, dtype=np.float32)

    def detach(self) -> Self:
        return self

    def float(self) -> Self:
        return self

    def cpu(self) -> Self:
        return self

    def numpy(self) -> np.ndarray:
        return self._samples


class _FakeCosyVoice2:
    sample_rate = 24000

    def __init__(self, model_dir: str, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self.model_dir = model_dir
        self.init_kwargs = kwargs
        self.inference_calls: list[tuple[tuple, dict]] = []

    def list_available_spks(self) -> list[str]:
        return ["my_zero_shot_spk"]

    def inference_instruct2(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        self.inference_calls.append((args, kwargs))
        yield {"tts_speech": np.array([[0.25, -0.5]], dtype=np.float32)}
        yield {"tts_speech": np.array([[0.75]], dtype=np.float32)}


class _FakeResponse:
    def __enter__(self):  # type: ignore[no-untyped-def]
        return self

    def __exit__(self, *_args):  # type: ignore[no-untyped-def]
        return False

    def raise_for_status(self) -> None:
        pass

    def iter_content(self, chunk_size):  # type: ignore[no-untyped-def]
        assert chunk_size == 4096
        audio = np.array([1, -2, 3], dtype=np.int16).tobytes()
        yield audio[:3]
        yield audio[3:]


def test_cosyvoice2_service_ignores_llm_training_metadata(mocker) -> None:  # type: ignore[no-untyped-def]
    """Fine-tuned checkpoints may include epoch and step beside model tensors."""
    original_load = mocker.Mock(
        return_value={"llm.weight": "tensor", "epoch": 3, "step": 4200}
    )
    torch = mocker.Mock(load=original_load)
    mocker.patch("scripts.serve_cosyvoice2_yue.import_module", return_value=torch)
    cosyvoice_module = mocker.Mock()
    cosyvoice_module.CosyVoice2.side_effect = lambda *_args, **_kwargs: torch.load(
        "/models/yue/llm.pt", map_location="cuda", weights_only=True
    )

    model = _load_cosyvoice2(cosyvoice_module, "/models/yue", fp16=True)

    assert model == {"llm.weight": "tensor"}
    assert torch.load is original_load
    original_load.assert_called_once_with(
        "/models/yue/llm.pt", map_location="cpu", weights_only=True
    )
    cosyvoice_module.CosyVoice2.assert_called_once_with("/models/yue", fp16=True)


def test_cosyvoice2_service_keeps_prompt_file_for_streaming_response(mocker) -> None:  # type: ignore[no-untyped-def]
    """The model receives a reusable path that is removed after streaming finishes."""
    prompt_paths: list[Path] = []
    model = mocker.Mock(sample_rate=24000)
    model.list_available_spks.return_value = ["my_zero_shot_spk"]

    def inference_instruct2(
        tts_text: str,
        instruct_text: str,
        prompt_wav: str,
        **kwargs,
    ):  # type: ignore[no-untyped-def]
        prompt_path = Path(prompt_wav)
        prompt_paths.append(prompt_path)
        assert prompt_path.read_bytes() == b"reference wav"
        assert tts_text == "歡迎嚟到理大。"
        assert instruct_text == "用粤语热情噉讲"
        assert kwargs == {"stream": True, "speed": 1.0, "text_frontend": False}
        yield {"tts_speech": _FakeSpeech([0.25, -0.5])}

    model.inference_instruct2.side_effect = inference_instruct2
    with TestClient(_build_app(model)) as client:
        response = client.post(
            "/inference_instruct2",
            data={
                "tts_text": "歡迎嚟到理大。",
                "instruct_text": "用粤语热情噉讲",
                "stream": "true",
                "speed": "1.0",
                "text_frontend": "false",
            },
            files={"prompt_wav": ("energetic.wav", b"reference wav", "audio/wav")},
        )

    expected = (np.array([0.25, -0.5]) * 32767).astype(np.int16).tobytes()
    assert response.status_code == 200
    assert response.content == expected
    assert len(prompt_paths) == 1
    assert not prompt_paths[0].exists()


def test_cosyvoice2_yue_forwards_voice_and_generation_parameters(mocker, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The backend forwards the configured reference voice and inference controls."""
    prompt_audio = tmp_path / "energetic.wav"
    prompt_audio.touch()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model = _FakeCosyVoice2("unused")
    model_cls = mocker.Mock(return_value=model)
    mocker.patch(
        "dimos.stream.audio.tts.node_cosyvoice2_yue._load_cosyvoice2",
        return_value=model_cls,
    )
    node = CosyVoice2YueTTSNode(
        prompt_audio=str(prompt_audio),
        model_dir=str(model_dir),
        instruct_text="用粤语开心噉讲",
        stream=True,
        speed=1.15,
        text_frontend=False,
        load_jit=True,
        load_trt=False,
        load_vllm=True,
        fp16=True,
        trt_concurrent=2,
    )

    try:
        events = list(node.iter_audio_events("歡迎嚟到理大!"))
    finally:
        node.dispose()

    model_cls.assert_called_once_with(
        str(model_dir),
        load_jit=True,
        load_trt=False,
        load_vllm=True,
        fp16=True,
        trt_concurrent=2,
    )
    assert model.inference_calls == [
        (
            ("歡迎嚟到理大!", "用粤语开心噉讲", str(prompt_audio)),
            {
                "zero_shot_spk_id": "",
                "stream": True,
                "speed": 1.15,
                "text_frontend": False,
            },
        )
    ]
    assert [event.data.tolist() for event in events] == [[0.25, -0.5], [0.75]]
    assert [event.sample_rate for event in events] == [24000, 24000]


def test_cosyvoice2_yue_uses_bundled_speaker_without_reference_audio(mocker, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The model's bundled speaker removes the need for a local prompt clip."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model = _FakeCosyVoice2("unused")
    model_cls = mocker.Mock(return_value=model)
    mocker.patch(
        "dimos.stream.audio.tts.node_cosyvoice2_yue._load_cosyvoice2",
        return_value=model_cls,
    )
    node = CosyVoice2YueTTSNode(model_dir=str(model_dir))

    try:
        events = list(node.iter_audio_events("歡迎嚟到理大"))
    finally:
        node.dispose()

    assert len(events) == 2
    assert model.inference_calls == [
        (
            ("歡迎嚟到理大", "用粤语以热情、亲切、有活力嘅语气说这句话", ""),
            {
                "zero_shot_spk_id": "my_zero_shot_spk",
                "stream": True,
                "speed": 1.0,
                "text_frontend": True,
            },
        )
    ]


def test_cosyvoice2_yue_requires_reference_audio() -> None:
    node = CosyVoice2YueTTSNode(prompt_audio="/missing/reference.wav")

    try:
        with pytest.raises(FileNotFoundError) as exc_info:
            node.prepare()
    finally:
        node.dispose()

    assert str(exc_info.value).endswith("/missing/reference.wav")


def test_cosyvoice2_yue_downloads_huggingface_model_id(mocker, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A Hugging Face model ID is resolved before CosyVoice sees the model path."""
    downloaded_model = tmp_path / "downloaded-model"
    hub = mocker.Mock()
    hub.snapshot_download.return_value = downloaded_model
    import_module = mocker.patch(
        "dimos.stream.audio.tts.node_cosyvoice2_yue.import_module",
        return_value=hub,
    )

    resolved = _resolve_model_dir(COSYVOICE2_YUE_MODEL)

    assert resolved == str(downloaded_model)
    import_module.assert_called_once_with("huggingface_hub")
    hub.snapshot_download.assert_called_once_with(repo_id=COSYVOICE2_YUE_MODEL)


def test_cosyvoice2_yue_http_uses_bundled_speaker_without_prompt(mocker) -> None:  # type: ignore[no-untyped-def]
    session = mocker.Mock()
    session.post.return_value = _FakeResponse()
    mocker.patch(
        "dimos.stream.audio.tts.node_cosyvoice2_yue.requests.Session",
        return_value=session,
    )
    node = CosyVoice2YueHTTPNode("http://localhost:50000/")

    try:
        events = list(node.iter_audio_events("歡迎嚟到理大"))
    finally:
        node.dispose()

    session.post.assert_called_once_with(
        "http://localhost:50000/inference_sft",
        data={
            "tts_text": "歡迎嚟到理大",
            "spk_id": "my_zero_shot_spk",
            "stream": "true",
            "speed": "1.0",
            "text_frontend": "true",
        },
        stream=True,
        timeout=None,
    )
    assert [event.data.tolist() for event in events] == [[1], [-2, 3]]
    assert [event.sample_rate for event in events] == [24000, 24000]
