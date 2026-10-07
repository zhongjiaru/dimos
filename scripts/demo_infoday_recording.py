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

"""Record one WAV from the default microphone: A starts, B saves and exits."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any
import wave

import numpy as np
from numpy.typing import NDArray
import sounddevice as sd  # type: ignore[import-untyped]

SAMPLE_RATE = 16000


def save_recording(chunks: list[NDArray[np.float32]], output_dir: Path) -> Path:
    audio = np.concatenate(chunks, axis=0)
    pcm = np.rint(np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"mic-test-{datetime.now():%Y%m%d-%H%M%S-%f}.wav"
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(pcm.tobytes())
    peak = float(np.max(np.abs(audio)))
    print(f"Saved: {path}", flush=True)
    print(f"Duration: {len(audio) / SAMPLE_RATE:.2f}s; peak: {peak:.4f}", flush=True)
    if peak == 0.0:
        print("Warning: the recording contains only silence.", flush=True)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/infoday-input-debug"))
    parser.add_argument("--button-device", default="Smart 2.4G Receiver")
    parser.add_argument("--start-keycode", type=int, default=112)
    parser.add_argument("--stop-keycode", type=int, default=117)
    args = parser.parse_args()
    if args.start_keycode == args.stop_keycode:
        parser.error("Start and stop keycodes must differ")
    xinput = shutil.which("xinput")
    stdbuf = shutil.which("stdbuf")
    if xinput is None or stdbuf is None:
        parser.error("xinput and stdbuf are required")

    chunks: list[NDArray[np.float32]] = []

    def capture(indata: NDArray[np.float32], _frames: int, _time: Any, status: Any) -> None:
        if status:
            print(f"Microphone status: {status}", file=sys.stderr, flush=True)
        chunks.append(indata.copy())

    monitor = subprocess.Popen(
        [stdbuf, "-oL", xinput, "test", args.button_device],
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    stream: Any | None = None
    try:
        print(f"Microphone: {sd.query_devices(kind='input')['name']}", flush=True)
        print(
            f"Ready: press A ({args.start_keycode}), speak, then B ({args.stop_keycode}).",
            flush=True,
        )
        assert monitor.stdout is not None
        for line in monitor.stdout:
            match = re.fullmatch(r"key\s+press\s+(\d+)\s*", line.strip())
            if match is None:
                continue
            keycode = int(match.group(1))
            if keycode == args.start_keycode and stream is None:
                stream = sd.InputStream(
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    blocksize=1024,
                    dtype=np.float32,
                    callback=capture,
                )
                stream.start()
                print("Recording... press B to save.", flush=True)
            elif keycode == args.stop_keycode and stream is not None:
                stream.stop()
                stream.close()
                stream = None
                if chunks:
                    save_recording(chunks, args.output_dir)
                    return 0
                print("No audio captured. Press A to try again.", flush=True)
        print("Button monitor exited before recording was saved.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Recording cancelled.", flush=True)
        return 130
    finally:
        if stream is not None:
            stream.stop()
            stream.close()
        if monitor.poll() is None:
            monitor.terminate()
        try:
            monitor.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            monitor.kill()
            monitor.wait()
        if monitor.stdout is not None:
            monitor.stdout.close()


if __name__ == "__main__":
    sys.exit(main())
