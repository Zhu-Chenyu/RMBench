"""
RMBench policy adapter for GR00T N1.6-3B fine-tuned with VLM memory annotations.

Uses a remote GR00T inference server (PolicyClient over ZMQ) so we don't have
to install flash-attn / GR00T deps inside the RMBench conda env. Start the
server first:

  /path/to/Isaac-GR00T/.venv/bin/python -m gr00t.eval.run_gr00t_server \\
      --model_path /path/to/checkpoint --embodiment_tag NEW_EMBODIMENT \\
      --device cuda:0 --port 5555 --no-strict
"""

import json
import os
import sys
from collections import deque

import numpy as np

# Make sure gr00t is importable for the PolicyClient
ISAAC_REPO = os.environ.get(
    "GR00T_REPO", "/home/luhr/chenyu/final_project/code/Isaac-GR00T"
)
if ISAAC_REPO not in sys.path:
    sys.path.insert(0, ISAAC_REPO)

from gr00t.policy.server_client import PolicyClient

# --- RMBench AgiLex bimanual layout (matches modality.json from convert_to_gr00t.py) -
LEFT_ARM = slice(0, 6)
LEFT_GRIPPER = slice(6, 7)
RIGHT_ARM = slice(7, 13)
RIGHT_GRIPPER = slice(13, 14)
LANGUAGE_KEY = "annotation.vlm.move.task_description"


class GR00TMemModel:
    def __init__(self, host: str = "localhost", port: int = 5555,
                 memory_text: str = "", action_steps: int = 16,
                 timeout_ms: int = 30000):
        self.client = PolicyClient(host=host, port=port,
                                   timeout_ms=timeout_ms, strict=False)
        if not self.client.ping():
            raise RuntimeError(
                f"GR00T server not reachable at {host}:{port}. "
                f"Start it with run_gr00t_server first."
            )
        self.memory_text = memory_text or "Memory: initial state, no objects moved."
        self.action_steps = action_steps
        self.obs_cache = deque(maxlen=1)

    # --- public RMBench-style hooks ----------------------------------------
    def update_obs(self, obs):
        self.obs_cache.append(obs)

    def reset_obs(self):
        self.obs_cache.clear()

    def set_memory(self, memory_text: str):
        self.memory_text = memory_text

    def get_action(self):
        """Return list of 14-D joint action vectors for the next horizon."""
        assert len(self.obs_cache) > 0, "obs_cache is empty"
        head_rgb, front_rgb, state14 = self.obs_cache[-1]

        # GR00T expects (B=1, T=1, H, W, 3) per camera and (B=1, T=1, D) per state slice
        observation = {
            "video": {
                "head":  head_rgb[None, None, ...],   # (1, 1, H, W, 3)
                "front": front_rgb[None, None, ...],
            },
            "state": {
                "left_arm":      state14[LEFT_ARM][None, None, ...].astype(np.float32),
                "left_gripper":  state14[LEFT_GRIPPER][None, None, ...].astype(np.float32),
                "right_arm":     state14[RIGHT_ARM][None, None, ...].astype(np.float32),
                "right_gripper": state14[RIGHT_GRIPPER][None, None, ...].astype(np.float32),
            },
            "language": {LANGUAGE_KEY: [[self.memory_text]]},
        }

        action_dict, _ = self.client.get_action(observation)
        # action_dict[k] shape: (1, T_action, D_k). Strip batch.
        la = action_dict["left_arm"][0]
        lg = action_dict["left_gripper"][0]
        ra = action_dict["right_arm"][0]
        rg = action_dict["right_gripper"][0]
        full = np.concatenate([la, lg, ra, rg], axis=-1)  # (T_action, 14)
        return [full[i] for i in range(min(self.action_steps, full.shape[0]))]


# ----------------------------------------------------------------------
# RMBench eval-script entry points
# ----------------------------------------------------------------------
def encode_obs(observation):
    head = observation["observation"]["head_camera"]["rgb"]
    front = observation["observation"]["front_camera"]["rgb"]
    state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
    return head, front, state


def get_model(usr_args):
    host = usr_args.get("server_host", "localhost")
    port = usr_args.get("server_port", 5555)
    action_steps = usr_args.get("action_steps", 16)
    mem_text = usr_args.get("memory_text", "")
    mem_json = usr_args.get("memory_text_path", None)
    task_name = usr_args.get("task_name", "")
    if mem_json and os.path.isfile(mem_json):
        with open(mem_json) as f:
            mp = json.load(f)
        mem_text = mp.get(task_name, mem_text)
    return GR00TMemModel(host=host, port=port, memory_text=mem_text,
                         action_steps=action_steps)


def eval(TASK_ENV, model, observation):
    obs = encode_obs(observation)
    if len(model.obs_cache) == 0:
        model.update_obs(obs)

    actions = model.get_action()
    for action in actions:
        TASK_ENV.take_action(action, action_type="qpos")
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        model.update_obs(obs)


def reset_model(model):
    model.reset_obs()
