"""Closed-loop RMBench deployment adapter for GR00T action checkpoints.

The RMBench evaluator imports three functions from this module:
`get_model`, `eval`, and `reset_model`. This adapter converts the live
RMBench observation dictionary into the modality format used by the GR00T
NEW_EMBODIMENT action checkpoints trained from RMBench data:

  video: head, front
  state/action: left_arm, left_gripper, right_arm, right_gripper
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any
from urllib import request

import numpy as np
from PIL import Image
import zmq


def _add_gr00t_to_path() -> None:
    gr00t_repo = os.environ.get("GR00T_REPO") or os.environ.get("ISAAC_GR00T_REPO")
    if gr00t_repo and gr00t_repo not in sys.path:
        sys.path.insert(0, gr00t_repo)


_add_gr00t_to_path()

from gr00t.data.embodiment_tags import EmbodimentTag  # noqa: E402
from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper  # noqa: E402
from gr00t.policy.server_client import PolicyClient  # noqa: E402


@dataclass
class GR00TRMBenchPolicy:
    policy: Gr00tSimPolicyWrapper
    execute_horizon: int
    head_camera_key: str
    front_camera_key: str
    language_mode: str
    memory_url: str | None = None
    memory_log_dir: str | None = None
    memory_max_chunks: int = 0
    memory_timeout_s: int = 20
    memory_language_max_chars: int = 0
    memory_long_budget: int = 1500
    memory_short_budget: int = 2300
    use_memory_as_language: bool = False
    prev_memory: str = ""
    episode_idx: int = -1
    chunk_idx: int = 0


def _enum_value(enum_cls: type, value: Any):
    if isinstance(value, enum_cls):
        return value
    value = str(value)
    if hasattr(enum_cls, value):
        return getattr(enum_cls, value)
    for item in enum_cls:
        if item.value == value:
            return item
    raise ValueError(f"Unknown {enum_cls.__name__}: {value}")


def _bool_arg(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _camera_rgb(observation: dict[str, Any], camera_key: str) -> np.ndarray:
    obs_block = observation.get("observation", {})
    candidates = [
        camera_key,
        camera_key.replace("_camera", ""),
        f"{camera_key}_camera",
    ]
    for key in candidates:
        if key in obs_block and isinstance(obs_block[key], dict) and "rgb" in obs_block[key]:
            arr = np.asarray(obs_block[key]["rgb"])
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            if arr.ndim != 3 or arr.shape[-1] != 3:
                raise ValueError(f"Camera {key}.rgb must be HxWx3, got {arr.shape}")
            return arr
    raise KeyError(f"Could not find RGB camera '{camera_key}' in observation")


def _joint_state(observation: dict[str, Any]) -> np.ndarray:
    joint_vec = observation.get("joint_action", {}).get("vector")
    if joint_vec is None:
        joint_vec = observation.get("joint_state", {}).get("vector")
    if joint_vec is None:
        raise KeyError("RMBench observation has no joint_action.vector or joint_state.vector")
    state = np.asarray(joint_vec, dtype=np.float32).reshape(-1)
    if state.size < 14:
        padded = np.zeros((14,), dtype=np.float32)
        padded[: state.size] = state
        state = padded
    elif state.size > 14:
        state = state[:14]
    return state


def _language(TASK_ENV, observation: dict[str, Any], mode: str) -> str:
    if mode == "empty":
        return ""
    if mode == "observation":
        return str(observation.get("instruction", ""))
    return str(TASK_ENV.get_instruction())


def _tag_body(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.S)
    return match.group(1).strip() if match else ""


def _clip_text(text: str, limit: int, tail: bool = False) -> str:
    text = " ".join(str(text).split())
    if limit <= 0 or len(text) <= limit:
        return text
    if tail:
        return "... " + text[-(limit - 4) :]
    return text[: limit - 4].rstrip() + " ..."


def _truncate_memory_for_action(model: GR00TRMBenchPolicy, text: str) -> str:
    max_total = int(model.memory_language_max_chars or 0)
    if max_total <= 0 or len(text) <= max_total:
        return text

    long = _tag_body(text, "LONG_TERM_MEMORY")
    short = _tag_body(text, "SHORT_TERM_MEMORY")
    if long or short:
        out = (
            "<MEMORY_CONTEXT>\n"
            f"<LONG_TERM_MEMORY>{_clip_text(long, model.memory_long_budget)}</LONG_TERM_MEMORY>\n"
            f"<SHORT_TERM_MEMORY>{_clip_text(short, model.memory_short_budget, tail=True)}</SHORT_TERM_MEMORY>\n"
            "</MEMORY_CONTEXT>"
        )
        if len(out) > max_total:
            out = out[: max_total - 4].rstrip() + " ..."
        return out
    return _clip_text(text, max_total)


def encode_obs(TASK_ENV, model: GR00TRMBenchPolicy, observation: dict[str, Any]) -> dict[str, Any]:
    state = _joint_state(observation)
    instruction = _language(TASK_ENV, observation, model.language_mode)
    if model.use_memory_as_language and model.prev_memory:
        memory = _truncate_memory_for_action(model, model.prev_memory)
        language = (
            "<TASK_INSTRUCTION>\n"
            f"{instruction}\n"
            "</TASK_INSTRUCTION>\n"
            f"{memory}"
        )
    else:
        language = instruction

    return {
        "video.head": _camera_rgb(observation, model.head_camera_key)[None, None, ...],
        "video.front": _camera_rgb(observation, model.front_camera_key)[None, None, ...],
        "state.left_arm": state[0:6][None, None, :],
        "state.left_gripper": state[6:7][None, None, :],
        "state.right_arm": state[7:13][None, None, :],
        "state.right_gripper": state[13:14][None, None, :],
        "annotation.vlm.move.task_description": (language,),
    }


def _image_b64(rgb: np.ndarray) -> str:
    image = Image.fromarray(rgb)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _generate_memory(
    TASK_ENV,
    model: GR00TRMBenchPolicy,
    observation: dict[str, Any],
    head_rgb: np.ndarray,
    front_rgb: np.ndarray,
) -> str | None:
    if not model.memory_url:
        return None
    if model.memory_max_chunks > 0 and model.chunk_idx >= model.memory_max_chunks:
        return None

    payload = {
        "head_b64": _image_b64(head_rgb),
        "front_b64": _image_b64(front_rgb),
        "prev_memory": _truncate_memory_for_action(model, model.prev_memory),
        "global_task": str(TASK_ENV.get_instruction()),
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        model.memory_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=model.memory_timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        memory = str(body.get("memory", "")).strip()
        if memory:
            model.prev_memory = memory
        return memory
    except Exception as exc:
        return f"[MEMORY_ERROR] {type(exc).__name__}: {exc}"


def _chunk_summary(chunk: np.ndarray, horizon: int) -> dict[str, Any]:
    executed = chunk[:horizon]
    diffs = np.diff(executed, axis=0) if len(executed) > 1 else np.zeros((0, executed.shape[-1]))
    return {
        "chunk_len": int(len(chunk)),
        "executed_horizon": int(horizon),
        "first_action": executed[0].round(5).tolist() if len(executed) else [],
        "last_executed_action": executed[-1].round(5).tolist() if len(executed) else [],
        "mean_abs_step_delta": float(np.mean(np.abs(diffs))) if diffs.size else 0.0,
        "max_abs_step_delta": float(np.max(np.abs(diffs))) if diffs.size else 0.0,
        "right_arm_first": executed[0, 7:13].round(5).tolist() if len(executed) else [],
        "right_arm_last": executed[-1, 7:13].round(5).tolist() if len(executed) else [],
        "right_arm_mean_abs_step_delta": float(np.mean(np.abs(diffs[:, 7:13]))) if diffs.size else 0.0,
        "right_arm_max_abs_step_delta": float(np.max(np.abs(diffs[:, 7:13]))) if diffs.size else 0.0,
    }


def _log_diagnostic(
    TASK_ENV,
    model: GR00TRMBenchPolicy,
    observation: dict[str, Any],
    memory: str | None,
    chunk: np.ndarray,
    horizon: int,
) -> None:
    if not model.memory_log_dir:
        return
    os.makedirs(model.memory_log_dir, exist_ok=True)
    path = os.path.join(model.memory_log_dir, f"episode_{model.episode_idx:04d}.jsonl")
    row = {
        "episode_idx": model.episode_idx,
        "chunk_idx": model.chunk_idx,
        "take_action_cnt_before": int(getattr(TASK_ENV, "take_action_cnt", -1)),
        "instruction": str(TASK_ENV.get_instruction()),
        "joint_state": _joint_state(observation).round(5).tolist(),
        "memory": memory,
        "memory_raw_chars": len(memory or ""),
        "memory_action_input_chars": len(_truncate_memory_for_action(model, model.prev_memory))
        if model.prev_memory
        else 0,
        "memory_logged": memory is not None,
        "action_summary": _chunk_summary(chunk, horizon),
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _log_event(
    TASK_ENV,
    model: GR00TRMBenchPolicy,
    event: str,
    extra: dict[str, Any] | None = None,
) -> None:
    if not model.memory_log_dir:
        return
    os.makedirs(model.memory_log_dir, exist_ok=True)
    path = os.path.join(model.memory_log_dir, f"episode_{model.episode_idx:04d}.events.jsonl")
    row = {
        "episode_idx": model.episode_idx,
        "chunk_idx": model.chunk_idx,
        "take_action_cnt": int(getattr(TASK_ENV, "take_action_cnt", -1)),
        "event": event,
        "time": time.time(),
    }
    if extra:
        row.update(extra)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _action_chunk_to_env(actions: dict[str, np.ndarray]) -> np.ndarray:
    left_arm = actions["action.left_arm"][0]
    left_gripper = actions["action.left_gripper"][0]
    right_arm = actions["action.right_arm"][0]
    right_gripper = actions["action.right_gripper"][0]
    return np.concatenate(
        [
            np.atleast_2d(left_arm),
            np.atleast_2d(left_gripper),
            np.atleast_2d(right_arm),
            np.atleast_2d(right_gripper),
        ],
        axis=-1,
    ).astype(np.float32)


def get_model(usr_args: dict[str, Any]) -> GR00TRMBenchPolicy:
    model_host = usr_args.get("model_host")
    if model_host:
        timeout_ms = int(usr_args.get("timeout_ms", 60000))
        policy = PolicyClient(
            host=str(model_host),
            port=int(usr_args.get("model_port", 5555)),
            timeout_ms=timeout_ms,
            strict=False,
        )
        policy.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        policy.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        return GR00TRMBenchPolicy(
            policy=policy,
            execute_horizon=int(usr_args.get("execute_horizon", 8)),
            head_camera_key=str(usr_args.get("head_camera_key", "head_camera")),
            front_camera_key=str(usr_args.get("front_camera_key", "front_camera")),
            language_mode=str(usr_args.get("language_mode", "instruction")),
            memory_url=usr_args.get("memory_url"),
            memory_log_dir=usr_args.get("memory_log_dir"),
            memory_max_chunks=int(usr_args.get("memory_max_chunks", 0)),
            memory_timeout_s=int(usr_args.get("memory_timeout_s", 20)),
            memory_language_max_chars=int(usr_args.get("memory_language_max_chars", 0)),
            memory_long_budget=int(usr_args.get("memory_long_budget", 1500)),
            memory_short_budget=int(usr_args.get("memory_short_budget", 2300)),
            use_memory_as_language=_bool_arg(usr_args.get("use_memory_as_language", False)),
        )

    model_path = usr_args.get("model_path")
    if not model_path:
        raise ValueError("GR00T eval requires --model_path /path/to/checkpoint or --model_host HOST")

    device = usr_args.get("device", "cuda")
    embodiment_tag = _enum_value(EmbodimentTag, usr_args.get("embodiment_tag", "NEW_EMBODIMENT"))
    base_policy = Gr00tPolicy(
        embodiment_tag=embodiment_tag,
        model_path=str(model_path),
        device=device,
        strict=True,
    )
    policy = Gr00tSimPolicyWrapper(base_policy, strict=True)
    return GR00TRMBenchPolicy(
        policy=policy,
        execute_horizon=int(usr_args.get("execute_horizon", 8)),
        head_camera_key=str(usr_args.get("head_camera_key", "head_camera")),
        front_camera_key=str(usr_args.get("front_camera_key", "front_camera")),
        language_mode=str(usr_args.get("language_mode", "instruction")),
        memory_url=usr_args.get("memory_url"),
        memory_log_dir=usr_args.get("memory_log_dir"),
        memory_max_chunks=int(usr_args.get("memory_max_chunks", 0)),
        memory_timeout_s=int(usr_args.get("memory_timeout_s", 20)),
        memory_language_max_chars=int(usr_args.get("memory_language_max_chars", 0)),
        memory_long_budget=int(usr_args.get("memory_long_budget", 1500)),
        memory_short_budget=int(usr_args.get("memory_short_budget", 2300)),
        use_memory_as_language=_bool_arg(usr_args.get("use_memory_as_language", False)),
    )


def eval(TASK_ENV, model: GR00TRMBenchPolicy, observation: dict[str, Any]):
    _log_event(TASK_ENV, model, "eval_start")
    head_rgb = _camera_rgb(observation, model.head_camera_key)
    front_rgb = _camera_rgb(observation, model.front_camera_key)
    memory = None
    if model.use_memory_as_language:
        memory_start = time.time()
        memory = _generate_memory(TASK_ENV, model, observation, head_rgb, front_rgb)
        _log_event(TASK_ENV, model, "memory_done", {"duration_s": time.time() - memory_start})
    obs = encode_obs(TASK_ENV, model, observation)
    action_start = time.time()
    _log_event(TASK_ENV, model, "action_start")
    action_dict, _info = model.policy.get_action(obs)
    _log_event(TASK_ENV, model, "action_done", {"duration_s": time.time() - action_start})
    chunk = _action_chunk_to_env(action_dict)
    horizon = min(model.execute_horizon, len(chunk))
    if not model.use_memory_as_language:
        memory_start = time.time()
        memory = _generate_memory(TASK_ENV, model, observation, head_rgb, front_rgb)
        _log_event(TASK_ENV, model, "memory_done", {"duration_s": time.time() - memory_start})
    _log_diagnostic(TASK_ENV, model, observation, memory, chunk, horizon)

    for action in chunk[:horizon]:
        TASK_ENV.take_action(action, action_type="qpos")
        if TASK_ENV.eval_success:
            break
    model.chunk_idx += 1


def reset_model(model: GR00TRMBenchPolicy):
    model.policy.reset()
    model.episode_idx += 1
    model.chunk_idx = 0
    model.prev_memory = ""
