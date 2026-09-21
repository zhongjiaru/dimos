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

"""Unit tests for UnitreeWebRTCConnection.

Pure-Python test suite with no hardware or network. Covers connect() error propagation,
aes_128_key forwarding, and the UNITREE_AES_128_KEY env var via GlobalConfig.
"""

import asyncio
from datetime import datetime, timezone
import json
import threading
from typing import Any
from unittest.mock import ANY, AsyncMock, MagicMock, call

from aiortc.mediastreams import MediaStreamError
from aiortc.stats import (
    RTCOutboundRtpStreamStats,
    RTCRemoteInboundRtpStreamStats,
    RTCStatsReport,
    RTCTransportStats,
)
import numpy as np
import pytest
from unitree_webrtc_connect.constants import (
    DATA_CHANNEL_TYPE,
    RTC_TOPIC,
    SPORT_CMD,
    WebRTCConnectionMethod,
)

from dimos.core.global_config import GlobalConfig
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.robot.unitree import connection as conn_mod
from dimos.robot.unitree.audio_track import GO2_AUDIO_SAMPLE_RATE
from dimos.robot.unitree.connection import UnitreeWebRTCConnection
from dimos.stream.audio.base import AudioEvent


def _stub_driver(connect_exc: Exception | None = None) -> MagicMock:
    """A LegionConnection instance double covering everything connect() touches."""
    driver = MagicMock(name="LegionConnection-instance")
    driver.connect = AsyncMock(side_effect=connect_exc)
    driver.disconnect = AsyncMock()
    driver.pc.connectionState = "connected"
    driver.pc.iceConnectionState = "completed"
    driver.datachannel.disableTrafficSaving = AsyncMock()
    driver.datachannel.set_decoder = MagicMock()
    driver.datachannel.pub_sub.publish_request_new = AsyncMock()
    return driver


def test_go2_driver_uses_max_bundle() -> None:
    driver = conn_mod.LegionConnection(
        conn_mod.WebRTCConnectionMethod.LocalSTA,
        ip="10.0.0.99",
    )

    config = driver.create_webrtc_configuration(None)

    assert config.bundlePolicy == conn_mod.RTCBundlePolicy.MAX_BUNDLE


def test_go2_driver_scopes_audio_first_sdk_factories(monkeypatch: pytest.MonkeyPatch) -> None:
    original_peer_connection = conn_mod.webrtc_driver.RTCPeerConnection
    original_audio_channel = conn_mod.webrtc_driver.WebRTCAudioChannel
    factories_seen: list[tuple[type[Any], type[Any]]] = []

    async def observe_factories(*_args: Any, **_kwargs: Any) -> None:
        factories_seen.append(
            (
                conn_mod.webrtc_driver.RTCPeerConnection,
                conn_mod.webrtc_driver.WebRTCAudioChannel,
            )
        )

    monkeypatch.setattr(conn_mod._LegionConnection, "init_webrtc", observe_factories)
    driver = conn_mod.LegionConnection(
        conn_mod.WebRTCConnectionMethod.LocalSTA,
        ip="10.0.0.99",
    )

    asyncio.run(driver.init_webrtc(ip="10.0.0.99"))

    assert factories_seen == [(conn_mod._AudioFirstPeerConnection, conn_mod._ExistingAudioChannel)]
    assert conn_mod.webrtc_driver.RTCPeerConnection is original_peer_connection
    assert conn_mod.webrtc_driver.WebRTCAudioChannel is original_audio_channel


def test_connect_failure_propagates_to_caller(monkeypatch: pytest.MonkeyPatch) -> None:
    """A driver connect failure must raise from the constructor, not hang."""
    driver = _stub_driver(connect_exc=RuntimeError("aes_128_key required (data2=3)"))
    monkeypatch.setattr(conn_mod, "LegionConnection", MagicMock(return_value=driver))

    with pytest.raises(RuntimeError, match="aes_128_key required"):
        UnitreeWebRTCConnection(ip="10.0.0.99")


@pytest.fixture
def built_connection(monkeypatch: pytest.MonkeyPatch) -> Any:
    """A live UnitreeWebRTCConnection over a stubbed driver, torn down (loop
    stopped, thread joined) unconditionally so a failed assert can't leak it."""
    driver = _stub_driver()
    monkeypatch.setattr(conn_mod, "LegionConnection", MagicMock(return_value=driver))

    conn = UnitreeWebRTCConnection(ip="10.0.0.99")
    try:
        yield conn, driver
    finally:
        conn.stop()


def test_connect_success_completes_setup(built_connection: Any) -> None:
    """Happy path: constructor returns after the setup sequence ran."""
    _conn, driver = built_connection

    driver.connect.assert_awaited_once()
    driver.datachannel.disableTrafficSaving.assert_awaited_once_with(True)
    driver.datachannel.pub_sub.publish_request_new.assert_awaited_once()


def test_connect_keeps_traffic_saving_enabled_when_lidar_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver = _stub_driver()
    monkeypatch.setattr(conn_mod, "LegionConnection", MagicMock(return_value=driver))

    connection = UnitreeWebRTCConnection(ip="10.0.0.99", lidar_enabled=False)
    try:
        driver.datachannel.disableTrafficSaving.assert_awaited_once_with(False)
    finally:
        connection.stop()


def test_video_track_end_completes_stream_without_callback_error(built_connection: Any) -> None:
    connection, driver = built_connection
    completed = threading.Event()
    subscription = connection.raw_video_stream().subscribe(on_completed=completed.set)
    callback = driver.video.add_track_callback.call_args.args[0]
    track = MagicMock()
    track.recv = AsyncMock(side_effect=MediaStreamError)

    try:
        future = asyncio.run_coroutine_threadsafe(callback(track), connection.loop)
        future.result(timeout=1.0)

        assert completed.is_set()
    finally:
        subscription.dispose()


def test_audio_output_attaches_and_queues_on_webrtc_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver = _stub_driver()
    sender = MagicMock(name="audio-sender")
    transceiver = MagicMock(sender=sender, currentDirection="sendrecv")
    driver.pc.addTrack.return_value = sender
    driver.pc.getTransceivers.return_value = [transceiver]
    track = MagicMock(name="QueuedGo2AudioTrack")
    track.enqueue.return_value = True
    track_factory = MagicMock(return_value=track)
    monkeypatch.setattr(conn_mod, "LegionConnection", MagicMock(return_value=driver))
    monkeypatch.setattr(conn_mod, "QueuedGo2AudioTrack", track_factory)
    add_track_thread: list[int] = []
    driver.pc.addTrack.side_effect = (
        lambda value: add_track_thread.append(threading.get_ident()) or sender
    )

    connection = UnitreeWebRTCConnection(ip="10.0.0.99", audio_output=True)
    try:
        event = AudioEvent(
            np.ones(960, dtype=np.int16),
            sample_rate=GO2_AUDIO_SAMPLE_RATE,
            timestamp=1.0,
            channels=1,
        )
        assert connection.audio_output_available()
        assert connection.enqueue_audio(event)
    finally:
        connection.stop()

    track_factory.assert_called_once_with()
    driver.pc.addTrack.assert_called_once_with(track)
    driver.audio.switchAudioChannel.assert_not_called()
    assert add_track_thread == [connection.thread.ident]
    track.enqueue.assert_called_once()
    track.stop.assert_called_once_with()


@pytest.mark.parametrize(
    ("connection_state", "ice_state"),
    [
        pytest.param("failed", "completed", id="peer-failed"),
        pytest.param("connected", "failed", id="ice-failed"),
        pytest.param("disconnected", "disconnected", id="disconnected"),
    ],
)
def test_audio_output_rejects_unhealthy_transport_without_toggling_audio_channel(
    monkeypatch: pytest.MonkeyPatch,
    connection_state: str,
    ice_state: str,
) -> None:
    driver = _stub_driver()
    driver.pc.connectionState = connection_state
    driver.pc.iceConnectionState = ice_state
    sender = MagicMock(name="audio-sender")
    driver.pc.addTrack.return_value = sender
    driver.pc.getTransceivers.return_value = [MagicMock(sender=sender, currentDirection="sendrecv")]
    monkeypatch.setattr(conn_mod, "LegionConnection", MagicMock(return_value=driver))

    connection = UnitreeWebRTCConnection(ip="10.0.0.99", audio_output=True)
    try:
        assert connection.audio_output_available() is False
    finally:
        connection.stop()

    driver.audio.switchAudioChannel.assert_not_called()


def _audio_sender_report(
    *,
    packets_sent: int,
    payload_bytes_sent: int,
) -> RTCStatsReport:
    now = datetime.now(timezone.utc)
    report = RTCStatsReport()
    report.add(
        RTCOutboundRtpStreamStats(
            timestamp=now,
            type="outbound-rtp",
            id="outbound-audio",
            ssrc=1,
            kind="audio",
            transportId="transport",
            packetsSent=packets_sent,
            bytesSent=payload_bytes_sent,
            trackId="audio-track",
        )
    )
    report.add(
        RTCRemoteInboundRtpStreamStats(
            timestamp=now,
            type="remote-inbound-rtp",
            id="remote-audio",
            ssrc=1,
            kind="audio",
            transportId="transport",
            packetsReceived=packets_sent,
            packetsLost=0,
            jitter=2,
            roundTripTime=0.01,
            fractionLost=0.0,
        )
    )
    report.add(
        RTCTransportStats(
            timestamp=now,
            type="transport",
            id="transport",
            packetsSent=packets_sent,
            packetsReceived=1,
            bytesSent=payload_bytes_sent,
            bytesReceived=100,
            iceRole="controlling",
            dtlsState="connected",
        )
    )
    return report


def test_audio_output_logs_rtp_stats_for_completed_send_window(
    monkeypatch: pytest.MonkeyPatch,
    mocker: Any,
) -> None:
    driver = _stub_driver()
    driver.pc.connectionState = "connected"
    driver.pc.iceConnectionState = "completed"
    sender = MagicMock(name="audio-sender")
    sender.getStats = AsyncMock(
        side_effect=[
            _audio_sender_report(packets_sent=100, payload_bytes_sent=10_000),
            _audio_sender_report(packets_sent=101, payload_bytes_sent=10_240),
        ]
    )
    transceiver = MagicMock(sender=sender, currentDirection="sendrecv")
    driver.pc.addTrack.return_value = sender
    driver.pc.getTransceivers.return_value = [transceiver]
    track = MagicMock(name="QueuedGo2AudioTrack")
    track.enqueue.return_value = True
    track.wait_drained = AsyncMock(return_value=True)
    track.take_pacing_diagnostics.return_value = {
        "frame_count": 2,
        "late_frame_count": 0,
        "max_lag_ms": 0.5,
        "max_catchup_streak": 0,
    }
    monkeypatch.setattr(conn_mod, "LegionConnection", MagicMock(return_value=driver))
    monkeypatch.setattr(conn_mod, "QueuedGo2AudioTrack", MagicMock(return_value=track))

    connection = UnitreeWebRTCConnection(ip="10.0.0.99", audio_output=True)
    log_info = mocker.patch.object(conn_mod.logger, "info")
    try:
        event = AudioEvent(
            np.ones(960, dtype=np.int16),
            sample_rate=GO2_AUDIO_SAMPLE_RATE,
            timestamp=1.0,
            channels=1,
        )

        assert connection.enqueue_audio(event) is True
        assert connection.wait_audio_drained(timeout=1.0) is True
    finally:
        connection.stop()

    log_info.assert_any_call(
        "Go2 WebRTC RTP send window",
        queue_drained=True,
        pcm_samples=960,
        pcm_duration_sec=0.02,
        expected_audio_packets=1,
        elapsed_sec=ANY,
        rtp_stats_available=True,
        rtp_packets_sent=1,
        rtp_payload_bytes_sent=240,
        rtp_packet_ratio=1.0,
        peer_connection_state="connected",
        ice_connection_state="completed",
        dtls_state="connected",
        remote_packets_received=1,
        remote_packets_lost=0,
        remote_fraction_lost=0.0,
        remote_jitter=2.0,
        round_trip_time=0.01,
        pacing_frames=2,
        pacing_late_frames=0,
        pacing_max_lag_ms=0.5,
        pacing_max_catchup_streak=0,
    )


@pytest.mark.parametrize(
    ("connection_options", "expected_call"),
    [
        pytest.param(
            {},
            call(
                RTC_TOPIC["WIRELESS_CONTROLLER"],
                data={"lx": 0.4, "ly": 1.5, "rx": -0.8, "ry": 0},
            ),
            id="joystick",
        ),
        pytest.param(
            {"velocity_api": True},
            call(
                RTC_TOPIC["SPORT_MOD"],
                data={
                    "header": {
                        "identity": {
                            "id": ANY,
                            "api_id": SPORT_CMD["Move"],
                        }
                    },
                    "parameter": json.dumps({"x": 1.5, "y": -0.4, "z": 0.8}),
                },
                msg_type=DATA_CHANNEL_TYPE["REQUEST"],
            ),
            id="velocity",
        ),
    ],
)
def test_move_api_toggle_sends_selected_wire_command(
    monkeypatch: pytest.MonkeyPatch,
    connection_options: dict[str, bool],
    expected_call: Any,
) -> None:
    driver = _stub_driver()
    monkeypatch.setattr(conn_mod, "LegionConnection", MagicMock(return_value=driver))
    twist = Twist(
        linear=Vector3(1.5, -0.4, 0.0),
        angular=Vector3(0.0, 0.0, 0.8),
    )

    connection = UnitreeWebRTCConnection(ip="10.0.0.99", **connection_options)
    try:
        driver.datachannel.pub_sub.publish_without_callback.reset_mock()
        assert connection.move(twist)
        assert driver.datachannel.pub_sub.publish_without_callback.call_args == expected_call
    finally:
        connection.stop_movement()
        connection.stop()


def test_stop_waits_for_disconnect_and_is_idempotent(built_connection: Any) -> None:
    """Concurrent stop calls wait for one peer close before stopping the loop."""
    connection, driver = built_connection
    disconnect_started = threading.Event()
    allow_disconnect = asyncio.Event()

    async def delayed_disconnect() -> None:
        disconnect_started.set()
        await allow_disconnect.wait()

    driver.disconnect.side_effect = delayed_disconnect
    first_stop = threading.Thread(target=connection.stop)
    second_stop = threading.Thread(target=connection.stop)

    first_stop.start()
    assert disconnect_started.wait(timeout=1.0)
    second_stop.start()
    assert first_stop.is_alive()
    assert second_stop.is_alive()

    connection.loop.call_soon_threadsafe(allow_disconnect.set)
    first_stop.join(timeout=1.0)
    second_stop.join(timeout=1.0)

    assert not first_stop.is_alive()
    assert not second_stop.is_alive()
    assert not connection.thread.is_alive()
    driver.disconnect.assert_awaited_once()


def test_stop_disconnects_when_stop_twist_fails(
    built_connection: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed final movement command must not leave the WebRTC peer open."""
    connection, driver = built_connection
    monkeypatch.setattr(
        connection,
        "_publish_movement",
        MagicMock(side_effect=RuntimeError("data channel closed")),
    )

    connection.stop()

    driver.disconnect.assert_awaited_once()
    assert not connection.thread.is_alive()


def test_liedown_uses_bounded_shutdown_request(
    built_connection: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing StandDown response must not block WebRTC teardown indefinitely."""
    connection, _driver = built_connection
    publish_request = MagicMock(return_value=True)
    monkeypatch.setattr(connection, "publish_request", publish_request)

    assert connection.liedown()
    publish_request.assert_called_once_with(
        RTC_TOPIC["SPORT_MOD"],
        {"api_id": SPORT_CMD["StandDown"]},
        timeout=1.0,
    )


@pytest.fixture
def stub_legion(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace LegionConnection with a mock and no-op connect() so __init__
    stays inside the aes_128_key resolution without dialing out."""
    monkeypatch.setattr(UnitreeWebRTCConnection, "connect", lambda self: None)
    legion = MagicMock(name="LegionConnection")
    monkeypatch.setattr(conn_mod, "LegionConnection", legion)
    return legion


def _aes_kwarg(legion: MagicMock) -> Any:
    """The aes_128_key passed to LegionConnection, or None if absent."""
    return legion.call_args.kwargs.get("aes_128_key")


def test_no_key_forwards_falsy(stub_legion: MagicMock) -> None:
    """No key → a falsy value reaches the driver, which treats it as no key."""
    UnitreeWebRTCConnection(ip="192.168.123.161")
    assert not _aes_kwarg(stub_legion)


def test_aes_key_forwarded_when_provided(stub_legion: MagicMock) -> None:
    """A provided key is forwarded verbatim to the driver."""
    UnitreeWebRTCConnection(ip="192.168.123.161", aes_128_key="aa" * 16)
    assert _aes_kwarg(stub_legion) == "aa" * 16


def test_empty_string_key_forwarded_as_falsy(stub_legion: MagicMock) -> None:
    """Empty-string key stays falsy → the driver treats it as no key."""
    UnitreeWebRTCConnection(ip="192.168.123.161", aes_128_key="")
    assert not _aes_kwarg(stub_legion)


def test_local_ap_connection_method_forwarded(stub_legion: MagicMock) -> None:
    UnitreeWebRTCConnection(
        ip=None,
        connection_method=WebRTCConnectionMethod.LocalAP,
    )
    assert stub_legion.call_args.args[0] is WebRTCConnectionMethod.LocalAP


def test_global_config_reads_unitree_aes_128_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The key enters via GlobalConfig, read from the UNITREE_AES_128_KEY env var."""
    monkeypatch.setenv("UNITREE_AES_128_KEY", "ee" * 16)
    assert GlobalConfig().unitree_aes_128_key == "ee" * 16
