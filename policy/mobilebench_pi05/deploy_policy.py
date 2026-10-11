# RMBench deploy hooks for MobileBench pi0.5 (memory-augmented), served over websocket.
#
# The PyTorch model runs in scripts/mobilebench/serve_rmbench.py (openpi fork, branch
# chenyu/mobilebench-v1.2) in its own venv; this adapter only packs observations with
# EXACTLY the pi0.5 baseline's camera layout (head -> cam_high, front -> cam_left_wrist,
# no right wrist) and tells the server when a new episode starts so the dual memory is
# reset. One server call = one memory write; the returned chunk's first pi0_step actions
# are executed open-loop, as for the baseline.

import json
import os

import numpy as np
from openpi_client import websocket_client_policy


class MemoryClient:
    def __init__(self, host, port, pi0_step):
        self.client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        self.pi0_step = pi0_step
        self.instruction = None
        self.new_episode = True

    def act(self, observation):
        to_chw = lambda im: np.ascontiguousarray(np.transpose(im, (2, 0, 1)))  # noqa: E731
        obs = {
            "images": {
                "cam_high": to_chw(observation["observation"]["head_camera"]["rgb"]),
                "cam_left_wrist": to_chw(observation["observation"]["front_camera"]["rgb"]),
            },
            "state": np.asarray(observation["joint_action"]["vector"], np.float32),
            "prompt": self.instruction,
            "reset": self.new_episode,
        }
        self.new_episode = False
        return self.client.infer(obs)["actions"]


def get_model(usr_args):
    port = int(os.environ.get("MB_SERVER_PORT", usr_args.get("server_port", 8765)))
    return MemoryClient("127.0.0.1", port, usr_args["pi0_step"])


# Optional per-action diagnostics (put_back_block success stages): set MB_DIAG_LOG=<file.jsonl>.
# Logs the task state check_success() works from: stage_id, press_cnt / press_flag, block pose
# vs. the centre and original (target) poses, and whether the right gripper is open.
DIAG_LOG = os.environ.get("MB_DIAG_LOG")


def _diag(TASK_ENV):
    blk = getattr(TASK_ENV, "block", None)
    if blk is None:
        return
    rec = {
        "ep": int(TASK_ENV.test_num),
        "step": int(TASK_ENV.take_action_cnt),
        "stage": int(getattr(TASK_ENV, "stage_id", -1)),
        "press_cnt": int(getattr(TASK_ENV, "press_cnt", -1)),
        "press_flag": bool(getattr(TASK_ENV, "press_flag", False)),
        "block": [round(float(x), 4) for x in blk.get_pose().p],
        "center": [round(float(x), 4) for x in np.asarray(TASK_ENV.center_pose)[:3]],
        "target": [round(float(x), 4) for x in np.asarray(TASK_ENV.target_pose)[:3]],
        "r_open": bool(TASK_ENV.is_right_gripper_open()),
        "success": bool(TASK_ENV.eval_success),
    }
    with open(DIAG_LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")


def eval(TASK_ENV, model, observation):
    if model.instruction is None:
        model.instruction = TASK_ENV.get_instruction()
        print(f"instruction: {model.instruction}")
    for action in model.act(observation)[: model.pi0_step]:
        TASK_ENV.take_action(action)
        if DIAG_LOG:
            _diag(TASK_ENV)


def reset_model(model):
    model.instruction = None
    model.new_episode = True
