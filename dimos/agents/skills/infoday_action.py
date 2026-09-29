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

# ruff: noqa: RUF001

from dataclasses import dataclass
import re
import threading
import time
from time import monotonic
from typing import Any, Literal
import unicodedata

from pydantic import Field
from unitree_webrtc_connect.constants import SPORT_CMD

from dimos.agents.annotation import skill
from dimos.agents.capabilities import CAP_MOVEMENT
from dimos.agents.skills.infoday_voice_answer_spec import InfodayVoiceAnswerSpec
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.robot.unitree.go2.connection_spec import GO2ConnectionSpec
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

InfodayAction = Literal[
    "wave",
    "stretch",
    "sit",
    "stand",
    "dance",
    "wiggle_hips",
    "finger_heart",
    "stop",
]


@dataclass(frozen=True)
class _ActionSpec:
    command: str
    duration_sec: float
    introduction: str
    spoken_name: str
    aliases: tuple[str, ...]


_ACTION_SPECS: dict[InfodayAction, _ActionSpec] = {
    "wave": _ActionSpec(
        "Hello",
        3.0,
        "好呀，我同你揮揮手，你睇住啦！",
        "揮手",
        (r"\bwave\b", r"揮手|挥手|打招呼"),
    ),
    "stretch": _ActionSpec(
        "Stretch",
        4.0,
        "好呀，我伸展一下先！",
        "伸展",
        (r"\bstretch\b", r"伸展|拉筋"),
    ),
    "sit": _ActionSpec(
        "Sit",
        3.0,
        "好呀，我而家坐低俾你睇！",
        "坐低",
        (r"\bsit\b", r"坐低|坐下"),
    ),
    "stand": _ActionSpec(
        "RiseSit",
        3.0,
        "好呀，我而家企返起身！",
        "企起身",
        (r"\bstand\b", r"企起身|企返起身|站起來|站起来"),
    ),
    "dance": _ActionSpec(
        "Dance1",
        8.0,
        "好呀，我跳隻舞俾你睇！",
        "跳舞",
        (r"\bdance\b", r"跳(?:返|一)?(?:隻|只|個|个|支|正)?舞"),
    ),
    "wiggle_hips": _ActionSpec(
        "WiggleHips",
        4.0,
        "好呀，睇下我扭下身先！",
        "扭身",
        (r"\bwiggle\b", r"扭身|扭下|扭屁股|擺動|摆动"),
    ),
    "finger_heart": _ActionSpec(
        "FingerHeart",
        5.0,
        "好呀，我送個心俾你！",
        "送心",
        (r"finger.?heart", r"比心|俾心|送心|愛心|爱心"),
    ),
    "stop": _ActionSpec(
        "StopMove",
        0.0,
        "好，我即刻停低。",
        "停低",
        (r"\bstop\b", r"停低|停止|唔好郁|不要动"),
    ),
}

_ACTION_COMPLETE = "完成啦！你想睇我做另一個動作，定係問下 EEE 嘅課程？"
_ACTION_FAILED = "唔好意思，呢個動作今次做唔到；你想試下揮手，定係問下 EEE 嘅課程？"
_SUGGESTED_ACTIONS: tuple[InfodayAction, ...] = ("wave", "stretch", "dance")
_ATTENTION_ACTIONS: tuple[InfodayAction, ...] = (
    "wave",
    "stretch",
    "finger_heart",
)
_CAPABILITY_OVERVIEW_PATTERN = r"what can you do|你識做咩|你识做咩|你會做咩|你会做什么"
_CAPABILITY_QUESTION_PATTERN = (
    r"\bcan you\b|識唔識|识不识|會唔會|会不会|你識|你识|你會|你会|得唔得|行唔行"
)
_DEMONSTRATION_PATTERN = (
    r"show|demonstrat|做(?:個|个|一次|下)|表演|示範|示范|俾我睇|畀我睇|給我看|给我看|"
    r"睇下|看看|來一個|来一个"
)
_ACTION_OFFER_PATTERN = r"你想|想唔想|想不想|可以揀|可以选|定係|還是|还是|或者|要唔要"
_ACTION_FOLLOWUP_PATTERNS = (
    r"^(?:好|好呀|好啊|可以|得|要|就呢個|就呢个|就這個|就这个)[!！。，, ]*$",
    r"^(?:我)?(?:想|要|可以)?(?:你)?(?:示範|示范|表演|做|試|试|睇|看)"
    r"(?:下|一下|一次|俾我睇|畀我睇|給我看|给我看)?[!！。，, ]*$",
)
_ROBOT_ACTION_INTENT_PATTERNS = (
    r"\b(?:move|walk|turn|follow|jump|navigate|come here)\b",
    r"\b(?:go|take me)\s+to\b",
    r"向前|向後|向后|後退|后退|轉左|转左|轉右|转右|跟住|跟隨|跟随",
    r"瞓低|躺下|翻身|後空翻|后空翻|跑|行去|過去|过去",
    r"導航|导航|帶我去|带我去|望下|影相|拍照|搵人|找人",
)


@dataclass(frozen=True)
class _ResolvedAction:
    action: InfodayAction | None
    message: str


class InfodayActionConfig(ModuleConfig):
    action_time_scale: float = Field(default=1.0, ge=0.0, le=2.0)


class InfodayActionSkill(Module):
    """Run a small, safe set of stationary Go2 actions for Info Day."""

    config: InfodayActionConfig
    go2: GO2ConnectionSpec
    voice_answer: InfodayVoiceAnswerSpec

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._motion_lock = threading.Lock()
        self._explicit_action_in_progress = False
        self._attention_busy_until = 0.0
        self._attention_action_index = 0

    @rpc
    def start_attention_action(self) -> str:
        """Start a short silent action while an Info Day answer is generated."""
        now = monotonic()
        with self._motion_lock:
            if self._explicit_action_in_progress or now < self._attention_busy_until:
                return "Skipped Info Day attention action: robot action already in progress"

            action = _ATTENTION_ACTIONS[self._attention_action_index % len(_ATTENTION_ACTIONS)]
            spec = _ACTION_SPECS[action]
            try:
                succeeded = self.go2.sport_command(SPORT_CMD[spec.command])
            except Exception as exc:
                logger.exception("InfoDay attention action failed", action=action)
                return f"Error starting Info Day attention action '{action}': {exc}"
            if not succeeded:
                logger.warning("InfoDay attention action rejected", action=action)
                return f"Info Day attention action '{action}' was rejected by the robot"

            self._attention_action_index += 1
            self._attention_busy_until = now + spec.duration_sec

        logger.info(
            "InfoDay attention action started",
            action=action,
            duration_sec=spec.duration_sec,
        )
        return f"Started Info Day attention action: {action}"

    @skill(uses=[CAP_MOVEMENT])
    def perform_robot_action(self, request: str) -> str:
        """Handle a robot capability/action request with matching speech and movement.

        Pass the user's original words. This tool determines whether the request is
        a capability question, a supported stationary demonstration, or an unsupported
        action. It never substitutes another action without a new explicit user request.
        It owns both the spoken response and physical command so they cannot conflict.

        Args:
            request: The user's complete original capability or action request.
        """
        clean_request = request.strip()
        resolved = _resolve_action_request(clean_request)
        if resolved.action is None:
            self.voice_answer.speak_message(resolved.message)
            return f"Answered robot action capability request: {clean_request or '<empty>'}"

        action = resolved.action
        spec = _ACTION_SPECS[action]
        with self._motion_lock:
            self._explicit_action_in_progress = True
        try:
            self.voice_answer.speak_message(resolved.message)
            try:
                if action == "stop":
                    self.go2.stop_movement()
                    succeeded = True
                else:
                    succeeded = self.go2.sport_command(SPORT_CMD[spec.command])
            except Exception as exc:
                logger.exception("InfoDay action command failed", action=action)
                self.voice_answer.speak_message(_ACTION_FAILED)
                return f"Error performing Info Day action '{action}': {exc}"

            if not succeeded:
                self.voice_answer.speak_message(_ACTION_FAILED)
                return f"Info Day action '{action}' was rejected by the robot"

            delay = spec.duration_sec * self.config.action_time_scale
            if delay:
                time.sleep(delay)
            self.voice_answer.speak_message(_ACTION_COMPLETE)
            return f"Completed Info Day action: {action}"
        finally:
            with self._motion_lock:
                self._explicit_action_in_progress = False


def _resolve_action_request(request: str) -> _ResolvedAction:
    normalized = _normalize_action_request(request)
    if re.search(_CAPABILITY_OVERVIEW_PATTERN, normalized):
        return _ResolvedAction(None, _capability_overview())

    wants_demo = bool(re.search(_DEMONSTRATION_PATTERN, normalized))
    matched_actions = _match_supported_actions(normalized)
    if "stop" in matched_actions:
        spec = _ACTION_SPECS["stop"]
        return _ResolvedAction("stop", spec.introduction)
    if not matched_actions:
        return _ResolvedAction(None, _unsupported_action_response())
    if len(matched_actions) > 1:
        names = "同".join(_ACTION_SPECS[action].spoken_name for action in matched_actions)
        if re.search(_CAPABILITY_QUESTION_PATTERN, normalized) and not wants_demo:
            return _ResolvedAction(None, f"我識{names}㗎。你想我示範邊一個？")
        return _ResolvedAction(
            None,
            f"為咗安全，我每次只做一個動作。你想我先做{names}入面邊一個？",
        )

    action = matched_actions[0]
    spec = _ACTION_SPECS[action]
    if re.search(_CAPABILITY_QUESTION_PATTERN, normalized) and not wants_demo:
        return _ResolvedAction(
            None,
            f"我識{spec.spoken_name}㗎。想唔想我而家做俾你睇？",
        )
    return _ResolvedAction(action, spec.introduction)


def is_robot_action_request(request: str) -> bool:
    """Return whether text expresses a robot capability or physical-action intent."""
    normalized = _normalize_action_request(request)
    has_capability_question = bool(re.search(_CAPABILITY_QUESTION_PATTERN, normalized))
    wants_demonstration = bool(re.search(_DEMONSTRATION_PATTERN, normalized))
    return bool(
        re.search(_CAPABILITY_OVERVIEW_PATTERN, normalized)
        or _match_supported_actions(normalized)
        or (has_capability_question and wants_demonstration)
        or any(re.search(pattern, normalized) for pattern in _ROBOT_ACTION_INTENT_PATTERNS)
    )


def resolve_offered_action_followup(previous_answer: str, request: str) -> str | None:
    """Resolve a short acceptance of the single action offered in the previous answer."""
    normalized_answer = _normalize_action_request(previous_answer)
    normalized_request = _normalize_action_request(request)
    if not re.search(_ACTION_OFFER_PATTERN, normalized_answer):
        return None
    offered_actions = _match_supported_actions(normalized_answer)
    if len(offered_actions) != 1:
        return None
    if not any(re.search(pattern, normalized_request) for pattern in _ACTION_FOLLOWUP_PATTERNS):
        return None
    return _ACTION_SPECS[offered_actions[0]].spoken_name


def _normalize_action_request(request: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        unicodedata.normalize("NFKC", request).casefold(),
    ).strip()


def _match_supported_actions(normalized_request: str) -> list[InfodayAction]:
    return [
        action
        for action, spec in _ACTION_SPECS.items()
        if any(re.search(alias, normalized_request) for alias in spec.aliases)
    ]


def _capability_overview() -> str:
    names = "、".join(
        spec.spoken_name for action, spec in _ACTION_SPECS.items() if action != "stop"
    )
    return f"我可以原地{names}。你想睇邊一個？"


def _unsupported_action_response() -> str:
    examples = "、".join(_ACTION_SPECS[action].spoken_name for action in _SUGGESTED_ACTIONS)
    return f"呢個動作我暫時未支援，不過我可以做其他動作俾你睇，例如原地{examples}。你想睇邊一個？"
