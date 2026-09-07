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

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]

from dimos.stream.audio.tts.node_canto_tts import CantoTTSNode


def test_canto_tts_node_materializes_default_onnx_model_once(tmp_path, mocker) -> None:  # type: ignore[no-untyped-def]
    """canto-tts output keeps the model's actual rate and channel count."""
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    blob = blobs / "weights"
    blob.write_bytes(b"model weights")
    snapshot = tmp_path / "snapshots" / "revision"
    model_dir = snapshot / "model"
    model_dir.mkdir(parents=True)
    (model_dir / "weights.data").symlink_to(blob)
    (snapshot / "browser_poc_manifest.json").write_text("{}")
    cache_dir = tmp_path / "dimos-cache"
    mocker.patch(
        "dimos.stream.audio.tts.node_canto_tts.resolve_onnx_model_dir",
        return_value=str(snapshot),
    )
    mocker.patch("dimos.stream.audio.tts.node_canto_tts._CANTO_TTS_CACHE_DIR", cache_dir)
    engine = mocker.Mock()

    def synthesize(_text: str, out_path: str) -> str:
        audio = np.array([[0.1, -0.1], [0.2, -0.2]], dtype=np.float32)
        sf.write(out_path, audio, 48000, subtype="FLOAT")
        return out_path

    engine.synthesize.side_effect = synthesize
    canto_tts = mocker.patch(
        "dimos.stream.audio.tts.node_canto_tts.CantoTTS",
        return_value=engine,
    )
    node = CantoTTSNode()

    try:
        node.prepare()
        first = list(node.iter_audio_events("多謝晒。"))
        second = list(node.iter_audio_events("你好。"))
    finally:
        node.dispose()

    materialized = cache_dir / "revision"
    canto_tts.assert_called_once_with(checkpoint=str(materialized), backend="onnx")
    assert (materialized / "model" / "weights.data").read_bytes() == b"model weights"
    assert not (materialized / "model" / "weights.data").is_symlink()
    assert [call.args[0] for call in engine.synthesize.call_args_list] == ["多謝晒。", "你好。"]
    assert len(first) == 1
    assert first[0].sample_rate == 48000
    assert first[0].channels == 2
    np.testing.assert_allclose(first[0].data, [[0.1, -0.1], [0.2, -0.2]])
    assert second[0].sample_rate == 48000


def test_canto_tts_node_forwards_checkpoint(mocker) -> None:  # type: ignore[no-untyped-def]
    engine = mocker.Mock()

    def synthesize(_text: str, out_path: str) -> str:
        sf.write(Path(out_path), np.array([0.1], dtype=np.float32), 48000, subtype="FLOAT")
        return out_path

    engine.synthesize.side_effect = synthesize
    canto_tts = mocker.patch(
        "dimos.stream.audio.tts.node_canto_tts.CantoTTS",
        return_value=engine,
    )
    node = CantoTTSNode(checkpoint="/models/canto-tts-nano")

    try:
        list(node.iter_audio_events("測試。"))
    finally:
        node.dispose()

    canto_tts.assert_called_once_with(
        checkpoint="/models/canto-tts-nano",
        backend="onnx",
    )
