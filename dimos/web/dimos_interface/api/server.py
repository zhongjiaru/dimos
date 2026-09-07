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


# Working FastAPI/Uvicorn Impl.

# Notes: Do not use simultaneously with Flask, this includes imports.
# Workers are not yet setup, as this requires a much more intricate
# reorganization. There appears to be possible signalling issues when
# opening up streams on multiple windows/reloading which will need to
# be fixed. Also note, Chrome only supports 6 simultaneous web streams,
# and its advised to test threading/worker performance with another
# browser like Safari.

# Fast Api & Uvicorn
import asyncio
import json
from pathlib import Path
from queue import Empty, Queue
import subprocess
import tempfile
from threading import Lock
import time

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import ffmpeg  # type: ignore[import-untyped]
import numpy as np
import reactivex as rx
from reactivex import operators as ops
from reactivex.disposable import SingleAssignmentDisposable
from sse_starlette.sse import EventSourceResponse
import uvicorn

from dimos.core.global_config import global_config
from dimos.stream.audio.base import AudioEvent
from dimos.web.edge_io import EdgeIO

# TODO: Resolve threading, start/stop stream functionality.


class FastAPIServer(EdgeIO):
    def __init__(  # type: ignore[no-untyped-def]
        self,
        dev_name: str = "FastAPI Server",
        edge_type: str = "Bidirectional",
        host: str | None = None,
        port: int = 5555,
        text_streams=None,
        audio_subject=None,
        audio_end_subject=None,
        **streams,
    ) -> None:
        super().__init__(dev_name, edge_type)
        self.app = FastAPI()
        self._server: uvicorn.Server | None = None

        # Add CORS middleware with more permissive settings for development
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # More permissive for development
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["*"],
        )

        self.port = port
        self.host = host if host is not None else global_config.listen_host
        BASE_DIR = Path(__file__).resolve().parent
        self.templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
        self.streams = streams
        self.active_streams = {}
        self.stream_locks = {key: Lock() for key in self.streams}
        self.stream_queues = {}  # type: ignore[var-annotated]
        self.stream_disposables = {}  # type: ignore[var-annotated]

        # Initialize text streams
        self.text_streams = text_streams or {}
        self.text_queues = {}  # type: ignore[var-annotated]
        self.text_disposables = {}
        self.text_clients = set()  # type: ignore[var-annotated]

        # Create a Subject for text queries
        self.query_subject = rx.subject.Subject()  # type: ignore[var-annotated]
        self.query_stream = self.query_subject.pipe(ops.share())
        self.audio_subject = audio_subject
        self.audio_end_subject = audio_end_subject

        for key in self.streams:
            if self.streams[key] is not None:
                self.active_streams[key] = self.streams[key].pipe(
                    ops.map(self.process_frame_fastapi), ops.share()
                )

        # Set up text stream subscriptions
        for key, stream in self.text_streams.items():
            if stream is not None:
                self.text_queues[key] = Queue(maxsize=100)
                disposable = stream.subscribe(
                    lambda text, k=key: self.text_queues[k].put(text) if text is not None else None,
                    lambda e, k=key: self.text_queues[k].put(None),
                    lambda k=key: self.text_queues[k].put(None),
                )
                self.text_disposables[key] = disposable
                self.disposables.add(disposable)

        self.setup_routes()

    def process_frame_fastapi(self, frame):  # type: ignore[no-untyped-def]
        """Convert frame to JPEG format for streaming."""
        import cv2

        _, buffer = cv2.imencode(".jpg", frame)
        return buffer.tobytes()

    def stream_generator(self, key):  # type: ignore[no-untyped-def]
        """Generate frames for a given video stream."""

        def generate():  # type: ignore[no-untyped-def]
            if key not in self.stream_queues:
                self.stream_queues[key] = Queue(maxsize=10)

            frame_queue = self.stream_queues[key]

            # Clear any existing disposable for this stream
            if key in self.stream_disposables:
                self.stream_disposables[key].dispose()

            disposable = SingleAssignmentDisposable()
            self.stream_disposables[key] = disposable
            self.disposables.add(disposable)

            if key in self.active_streams:
                with self.stream_locks[key]:
                    # Clear the queue before starting new subscription
                    while not frame_queue.empty():
                        try:
                            frame_queue.get_nowait()
                        except Empty:
                            break

                    disposable.disposable = self.active_streams[key].subscribe(
                        lambda frame: frame_queue.put(frame) if frame is not None else None,
                        lambda e: frame_queue.put(None),
                        lambda: frame_queue.put(None),
                    )

            try:
                while True:
                    try:
                        frame = frame_queue.get(timeout=1)
                        if frame is None:
                            break
                        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
                    except Empty:
                        # Instead of breaking, continue waiting for new frames
                        continue
            finally:
                if key in self.stream_disposables:
                    self.stream_disposables[key].dispose()

        return generate

    def create_video_feed_route(self, key):  # type: ignore[no-untyped-def]
        """Create a video feed route for a specific stream."""

        async def video_feed():  # type: ignore[no-untyped-def]
            return StreamingResponse(
                self.stream_generator(key)(),  # type: ignore[no-untyped-call]
                media_type="multipart/x-mixed-replace; boundary=frame",
            )

        return video_feed

    async def text_stream_generator(self, key):  # type: ignore[no-untyped-def]
        """Generate SSE events for text stream."""
        client_id = id(object())
        self.text_clients.add(client_id)

        try:
            while True:
                if key not in self.text_queues:
                    yield {"event": "ping", "data": ""}
                    await asyncio.sleep(0.1)
                    continue

                try:
                    text = self.text_queues[key].get_nowait()
                    if text is not None:
                        yield {"event": "message", "id": key, "data": text}
                    else:
                        break
                except Empty:
                    yield {"event": "ping", "data": ""}
                    await asyncio.sleep(0.1)
        finally:
            self.text_clients.remove(client_id)

    @staticmethod
    def _decode_audio(raw: bytes) -> tuple[np.ndarray | None, int | None]:
        """Convert an uploaded audio file into mono 16-kHz float32 PCM."""
        try:
            # MP4/M4A metadata can require seeking, which is impossible when
            # ffmpeg reads stdin. A temporary input also lets ffmpeg probe all
            # supported upload formats. Headerless output remains pipe-safe.
            with tempfile.TemporaryDirectory() as temp_dir:
                input_path = Path(temp_dir) / "upload"
                input_path.write_bytes(raw)
                out, _ = (
                    ffmpeg.input(str(input_path))
                    .output(
                        "pipe:1",
                        format="f32le",
                        acodec="pcm_f32le",
                        ac=1,
                        ar="16000",
                        loglevel="quiet",
                    )
                    .run(capture_stdout=True, capture_stderr=True)
                )
            audio = np.frombuffer(out, dtype="<f4").copy()
            if audio.size == 0:
                raise ValueError("ffmpeg decoded no audio samples")
            return audio, 16000
        except Exception as exc:
            print(f"ffmpeg decoding failed: {exc}")
            return None, None

    def setup_routes(self) -> None:
        """Set up FastAPI routes."""

        @self.app.get("/streams")
        async def get_streams():  # type: ignore[no-untyped-def]
            """Get list of available video streams"""
            return {"streams": list(self.streams.keys())}

        @self.app.get("/text_streams")
        async def get_text_streams():  # type: ignore[no-untyped-def]
            """Get list of available text streams"""
            return {"streams": list(self.text_streams.keys())}

        @self.app.get("/", response_class=HTMLResponse)
        async def index(request: Request):  # type: ignore[no-untyped-def]
            stream_keys = list(self.streams.keys())
            text_stream_keys = list(self.text_streams.keys())
            # Starlette 0.38+: request is 1st positional arg, not in context dict.
            return self.templates.TemplateResponse(
                request,
                "index_fastapi.html",
                {
                    "stream_keys": stream_keys,
                    "text_stream_keys": text_stream_keys,
                    "has_voice": self.audio_subject is not None,
                },
            )

        @self.app.post("/submit_query")
        async def submit_query(query: str = Form(...)):  # type: ignore[no-untyped-def]
            # Using Form directly as a dependency ensures proper form handling
            try:
                if query:
                    # Emit the query through our Subject
                    self.query_subject.on_next(query)
                    return JSONResponse({"success": True, "message": "Query received"})
                return JSONResponse({"success": False, "message": "No query provided"})
            except Exception as e:
                # Ensure we always return valid JSON even on error
                return JSONResponse(
                    status_code=500,
                    content={"success": False, "message": f"Server error: {e!s}"},
                )

        @self.app.post("/upload_audio")
        async def upload_audio(file: UploadFile = File(...)):  # type: ignore[no-untyped-def]
            """Handle audio upload from the browser."""
            if self.audio_subject is None:
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "message": "Voice input not configured"},
                )

            try:
                data = await file.read()
                audio_np, sr = self._decode_audio(data)
                if audio_np is None or sr is None:
                    return JSONResponse(
                        status_code=400,
                        content={"success": False, "message": "Unable to decode audio"},
                    )

                event = AudioEvent(
                    data=audio_np,
                    sample_rate=sr,
                    timestamp=time.time(),
                    channels=1 if audio_np.ndim == 1 else audio_np.shape[1],
                )

                # Push to reactive stream
                self.audio_subject.on_next(event)
                if self.audio_end_subject is not None:
                    self.audio_end_subject.on_next(None)
                print(f"Received audio - {event.data.shape[0] / sr:.2f} s, {sr} Hz")
                return {"success": True}
            except Exception as e:
                print(f"Failed to process uploaded audio: {e}")
                return JSONResponse(status_code=500, content={"success": False, "message": str(e)})

        @self.app.websocket("/stream_audio")
        async def stream_audio(websocket: WebSocket):  # type: ignore[no-untyped-def]
            """Receive browser microphone PCM chunks for streaming speech input."""
            if self.audio_subject is None:
                await websocket.close(code=1008, reason="Voice input not configured")
                return

            await websocket.accept()
            sample_rate = 48000
            channels = 1
            try:
                while True:
                    message = await websocket.receive()
                    if message.get("type") == "websocket.disconnect":
                        break
                    if message.get("text") is not None:
                        payload = json.loads(message["text"])
                        event_type = payload.get("type")
                        if event_type == "start":
                            sample_rate = int(payload.get("sample_rate") or sample_rate)
                            channels = int(payload.get("channels") or channels)
                            await websocket.send_json({"success": True, "type": "started"})
                        elif event_type == "stop":
                            if self.audio_end_subject is not None:
                                self.audio_end_subject.on_next(None)
                            await websocket.send_json({"success": True, "type": "stopped"})
                            break
                    elif message.get("bytes") is not None:
                        audio = np.frombuffer(message["bytes"], dtype=np.float32).copy()
                        if audio.size == 0:
                            continue
                        self.audio_subject.on_next(
                            AudioEvent(
                                data=audio,
                                sample_rate=sample_rate,
                                timestamp=time.time(),
                                channels=channels,
                            )
                        )
            except Exception as exc:
                print(f"Streaming audio websocket closed: {exc}")
                if self.audio_end_subject is not None:
                    self.audio_end_subject.on_next(None)

        # Unitree API endpoints
        @self.app.get("/unitree/status")
        async def unitree_status():  # type: ignore[no-untyped-def]
            """Check the status of the Unitree API server"""
            return JSONResponse({"status": "online", "service": "unitree"})

        @self.app.post("/unitree/command")
        async def unitree_command(request: Request):  # type: ignore[no-untyped-def]
            """Process commands sent from the terminal frontend"""
            try:
                data = await request.json()
                command_text = data.get("command", "")

                # Emit the command through the query_subject
                self.query_subject.on_next(command_text)

                response = {
                    "success": True,
                    "command": command_text,
                    "result": f"Processed command: {command_text}",
                }

                return JSONResponse(response)
            except Exception as e:
                print(f"Error processing command: {e!s}")
                return JSONResponse(
                    status_code=500,
                    content={"success": False, "message": f"Error processing command: {e!s}"},
                )

        @self.app.get("/text_stream/{key}")
        async def text_stream(key: str):  # type: ignore[no-untyped-def]
            if key not in self.text_streams:
                raise HTTPException(status_code=404, detail=f"Text stream '{key}' not found")
            return EventSourceResponse(self.text_stream_generator(key))  # type: ignore[no-untyped-call]

        for key in self.streams:
            self.app.get(f"/video_feed/{key}")(self.create_video_feed_route(key))  # type: ignore[no-untyped-call]

    @staticmethod
    def _ensure_certs(certs_dir: Path) -> tuple[str, str]:
        """Return (cert_path, key_path), generating self-signed certs if needed.
        HTTPS is required by browsers for sensor APIs (DeviceOrientation)"""
        cert_path = certs_dir / "cert.pem"
        key_path = certs_dir / "key.pem"

        if cert_path.exists() and key_path.exists():
            return str(cert_path), str(key_path)

        certs_dir.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [
                "openssl",
                "req",
                "-x509",
                "-newkey",
                "rsa:2048",
                "-keyout",
                str(key_path),
                "-out",
                str(cert_path),
                "-days",
                "365",
                "-nodes",
                "-subj",
                "/CN=localhost",
            ],
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to generate certificates: {result.stderr.decode()}")
        return str(cert_path), str(key_path)

    def run(self, ssl: bool = False, ssl_certs_dir: Path | str | None = None) -> None:
        ssl_certfile = None
        ssl_keyfile = None

        if ssl:
            if ssl_certs_dir is None:
                raise ValueError("ssl_certs_dir is required when ssl=True")
            ssl_certfile, ssl_keyfile = self._ensure_certs(Path(ssl_certs_dir))

        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="error",  # Reduce verbosity
            ssl_certfile=ssl_certfile,
            ssl_keyfile=ssl_keyfile,
        )
        self._server = uvicorn.Server(config)
        self._server.run()

    def shutdown(self) -> None:
        if self._server is not None:
            self._server.should_exit = True


if __name__ == "__main__":
    server = FastAPIServer()
    server.run()
