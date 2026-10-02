# RMBench deploy hooks for π0.5 trained with pi05_rmbench_put_back_block_lora.
#
# encode_obs feeds [head_camera, front_camera] -- the two static cameras that
# aloha-agilex declares and that the GR00T conversion turned into
# observation.images.{head,front}. The wrist cameras are deliberately NOT used:
# the model never saw them during training. See pi_model.py for the full rationale.

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pi_model import PI0  # noqa: E402


def encode_obs(observation):
    input_rgb_arr = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["front_camera"]["rgb"],
    ]
    input_state = observation["joint_action"]["vector"]
    return input_rgb_arr, input_state


def get_model(usr_args):
    return PI0(
        usr_args["train_config_name"],
        usr_args["model_name"],
        usr_args["checkpoint_id"],
        usr_args["pi0_step"],
    )


def eval(TASK_ENV, model, observation):
    if model.observation_window is None:
        model.set_language(TASK_ENV.get_instruction())

    input_rgb_arr, input_state = encode_obs(observation)
    model.update_observation_window(input_rgb_arr, input_state)

    actions = model.get_action()[: model.pi0_step]
    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        input_rgb_arr, input_state = encode_obs(observation)
        model.update_observation_window(input_rgb_arr, input_state)


def reset_model(model):
    model.reset_obsrvationwindows()
