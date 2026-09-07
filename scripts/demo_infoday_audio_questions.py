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

"""Exercise the running Info Day blueprint with prerecorded questions."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import mimetypes
from pathlib import Path
import queue
import re
import sys
import time
from typing import Any

import requests

from dimos.core.global_config import global_config
from dimos.core.transport import PubSubTransport
from dimos.core.transport_factory import make_transport
from dimos.stream.audio.base import AudioEvent

SUPPORTED_AUDIO_SUFFIXES = frozenset(
    {".aac", ".flac", ".m4a", ".mp3", ".mp4", ".ogg", ".wav", ".webm"}
)
DEFAULT_WEB_URL = "http://localhost:5555"
DEFAULT_UPLOAD_TIMEOUT_SEC = 30.0
DEFAULT_TRANSCRIPT_TIMEOUT_SEC = 60.0
DEFAULT_ANSWER_TIMEOUT_SEC = 120.0
DEFAULT_AUDIO_IDLE_SEC = 8.0


class PipelineTimeoutError(TimeoutError):
    """Raised when one stage of the Info Day pipeline does not produce output."""


class PipelineObserver:
    """Observe ASR text and generated answer audio on DimOS transports."""

    def __init__(self) -> None:
        self._transcripts: queue.Queue[str] = queue.Queue()
        self._answers: queue.Queue[str] = queue.Queue()
        self._audio: queue.Queue[AudioEvent] = queue.Queue()
        self._transcript_transport: PubSubTransport[Any] = make_transport("/infoday_input")
        self._answer_transport: PubSubTransport[Any] = make_transport("/infoday_answer")
        self._audio_transport: PubSubTransport[Any] = make_transport("/operator_audio", AudioEvent)
        self._unsubscribe: list[Callable[[], None]] = []

    def __enter__(self) -> PipelineObserver:
        self._unsubscribe = [
            self._transcript_transport.subscribe(self._transcripts.put),
            self._answer_transport.subscribe(self._answers.put),
            self._audio_transport.subscribe(self._audio.put),
        ]
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: object | None,
    ) -> None:
        for unsubscribe in self._unsubscribe:
            unsubscribe()
        self._audio_transport.stop()
        self._answer_transport.stop()
        self._transcript_transport.stop()

    def reset(self) -> None:
        _drain_queue(self._transcripts)
        _drain_queue(self._answers)
        _drain_queue(self._audio)

    def wait_for_transcript(self, timeout_sec: float) -> str:
        try:
            return self._transcripts.get(timeout=timeout_sec)
        except queue.Empty as exc:
            raise PipelineTimeoutError(
                f"ASR did not publish to /infoday_input within {timeout_sec:g}s"
            ) from exc

    def wait_for_answer_text(self, timeout_sec: float) -> str:
        try:
            return self._answers.get(timeout=timeout_sec)
        except queue.Empty as exc:
            raise PipelineTimeoutError(
                f"The LLM did not publish to /infoday_answer within {timeout_sec:g}s"
            ) from exc

    def wait_for_answer_audio(
        self,
        *,
        first_chunk_timeout_sec: float,
        idle_sec: float,
    ) -> tuple[int, float]:
        try:
            first = self._audio.get(timeout=first_chunk_timeout_sec)
        except queue.Empty as exc:
            raise PipelineTimeoutError(
                "No answer audio was published to /operator_audio within "
                f"{first_chunk_timeout_sec:g}s"
            ) from exc

        chunks = 1
        duration_sec = _audio_duration(first)
        while True:
            try:
                event = self._audio.get(timeout=idle_sec)
            except queue.Empty:
                return chunks, duration_sec
            chunks += 1
            duration_sec += _audio_duration(event)


def _drain_queue(items: queue.Queue[Any]) -> None:
    while True:
        try:
            items.get_nowait()
        except queue.Empty:
            return


def _audio_duration(event: AudioEvent) -> float:
    if event.sample_rate <= 0 or event.channels <= 0:
        return 0.0
    return event.data.size / (event.sample_rate * event.channels)


def _natural_sort_key(path: Path) -> tuple[str, ...]:
    return tuple(
        f"{int(part):020d}" if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", path.name)
    )


def resolve_audio_files(inputs: Sequence[Path]) -> list[Path]:
    files: list[Path] = []
    for input_path in inputs:
        if input_path.is_dir():
            files.extend(
                path
                for path in input_path.iterdir()
                if path.is_file() and path.suffix.casefold() in SUPPORTED_AUDIO_SUFFIXES
            )
        elif input_path.is_file():
            if input_path.suffix.casefold() not in SUPPORTED_AUDIO_SUFFIXES:
                raise ValueError(f"Unsupported audio file: {input_path}")
            files.append(input_path)
        else:
            raise FileNotFoundError(f"Audio input does not exist: {input_path}")

    unique_files = sorted(set(files), key=_natural_sort_key)
    if not unique_files:
        suffixes = ", ".join(sorted(SUPPORTED_AUDIO_SUFFIXES))
        raise ValueError(f"No supported audio files found (supported: {suffixes})")
    return unique_files


def check_web_input(session: requests.Session, web_url: str, timeout_sec: float) -> None:
    response = session.get(f"{web_url}/text_streams", timeout=timeout_sec)
    response.raise_for_status()


def upload_audio(
    session: requests.Session,
    web_url: str,
    path: Path,
    timeout_sec: float,
) -> None:
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    with path.open("rb") as audio_file:
        response = session.post(
            f"{web_url}/upload_audio",
            files={"file": (path.name, audio_file, content_type)},
            timeout=timeout_sec,
        )
    response.raise_for_status()
    payload = response.json()
    if not payload.get("success"):
        raise RuntimeError(f"WebInput rejected {path.name}: {payload}")


def run_questions(args: argparse.Namespace) -> int:
    if args.transport is not None:
        global_config.update(transport=args.transport)

    audio_files = resolve_audio_files(args.inputs)
    session = requests.Session()
    try:
        check_web_input(session, args.web_url, args.upload_timeout)
    except requests.RequestException as exc:
        session.close()
        raise RuntimeError(
            f"Cannot reach WebInput at {args.web_url}. Start the Info Day blueprint first."
        ) from exc

    print(f"Found {len(audio_files)} audio question(s); transport={global_config.transport}")
    failures = 0
    passed = 0
    try:
        with PipelineObserver() as observer:
            for index, path in enumerate(audio_files, start=1):
                observer.reset()
                started_at = time.monotonic()
                print(f"\n[{index}/{len(audio_files)}] Uploading {path.name}")
                try:
                    upload_audio(session, args.web_url, path, args.upload_timeout)
                    transcript = observer.wait_for_transcript(args.transcript_timeout)
                    print(f"  ASR: {transcript}")
                    answer_text = observer.wait_for_answer_text(args.answer_timeout)
                    print(f"  LLM answer: {answer_text}")
                    chunks, audio_duration = observer.wait_for_answer_audio(
                        first_chunk_timeout_sec=args.answer_timeout,
                        idle_sec=args.audio_idle,
                    )
                    elapsed = time.monotonic() - started_at
                    print(
                        "  Answer audio complete: "
                        f"{chunks} chunk(s), {audio_duration:.2f}s audio, {elapsed:.2f}s elapsed"
                    )
                    passed += 1
                    if args.interactive:
                        input("  Press Enter after verifying the robot's answer...")
                except (PipelineTimeoutError, requests.RequestException, RuntimeError) as exc:
                    failures += 1
                    print(f"  FAILED: {exc}", file=sys.stderr)
                    if not args.keep_going:
                        break
    finally:
        session.close()

    print(f"\nResult: {passed} passed, {failures} failed")
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Upload prerecorded questions through WebInput and wait for the running "
            "Info Day blueprint to publish each spoken answer."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Audio files and/or directories containing audio files.",
    )
    parser.add_argument(
        "--web-url",
        default=DEFAULT_WEB_URL,
        help=f"WebInput base URL (default: {DEFAULT_WEB_URL}).",
    )
    parser.add_argument(
        "--transport",
        choices=("lcm", "zenoh"),
        help="Transport used by the running blueprint (default: resolved DimOS config).",
    )
    parser.add_argument(
        "--upload-timeout",
        type=float,
        default=DEFAULT_UPLOAD_TIMEOUT_SEC,
        help=f"Audio upload timeout in seconds (default: {DEFAULT_UPLOAD_TIMEOUT_SEC:g}).",
    )
    parser.add_argument(
        "--transcript-timeout",
        type=float,
        default=DEFAULT_TRANSCRIPT_TIMEOUT_SEC,
        help=f"ASR result timeout in seconds (default: {DEFAULT_TRANSCRIPT_TIMEOUT_SEC:g}).",
    )
    parser.add_argument(
        "--answer-timeout",
        type=float,
        default=DEFAULT_ANSWER_TIMEOUT_SEC,
        help=f"First answer-audio timeout in seconds (default: {DEFAULT_ANSWER_TIMEOUT_SEC:g}).",
    )
    parser.add_argument(
        "--audio-idle",
        type=float,
        default=DEFAULT_AUDIO_IDLE_SEC,
        help=(
            "Seconds without /operator_audio before an answer is considered complete "
            f"(default: {DEFAULT_AUDIO_IDLE_SEC:g})."
        ),
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Pause after each answer so a human can verify what the robot said.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue with the remaining files after a failed question.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in ("upload_timeout", "transcript_timeout", "answer_timeout", "audio_idle"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be greater than zero")
    try:
        return run_questions(args)
    except (FileNotFoundError, ValueError, RuntimeError, requests.RequestException) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
