# π0.5 policy adapter for RMBench closed-loop eval, built against the OFFICIAL openpi.
#
# Differences from RMBench's bundled policy/pi05/pi_model.py:
#
#  1. Official openpi's `create_trained_policy` has no `robotwin_repo_id` kwarg. It
#     resolves norm stats from the checkpoint's own assets/ using the train config's
#     asset_id, so nothing extra is needed.
#
#  2. CAMERA MAPPING. This is the part that matters. The RMBench adapter packs
#     cam_high=head, cam_left_wrist=left_wrist, cam_right_wrist=right_wrist. But
#     pi05_rmbench_put_back_block_lora was TRAINED with:
#         cam_high        <- observation.images.head   (head_camera)
#         cam_left_wrist  <- observation.images.front  (front_camera)
#         cam_right_wrist <- absent, zero-filled and masked off by AlohaInputs
#     Feeding the wrist cameras here instead would put front-view-trained weights on
#     wrist pixels and un-mask a slot that was always black during training. So we
#     reproduce the training layout exactly and simply omit the right wrist.
#
#     (aloha-agilex/config.yml declares exactly two static cameras, head_camera and
#     front_camera, which is where the GR00T conversion's "head"/"front" came from.)

import os

import numpy as np
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


class PI0:
    def __init__(self, train_config_name, model_name, checkpoint_id, pi0_step):
        self.train_config_name = train_config_name
        self.model_name = model_name
        self.checkpoint_id = checkpoint_id
        self.pi0_step = pi0_step

        ckpt_dir = os.environ.get("PI05_CHECKPOINT_DIR")
        if not ckpt_dir:
            raise RuntimeError("set PI05_CHECKPOINT_DIR to the checkpoint step directory")

        config = _config.get_config(train_config_name)
        self.policy = _policy_config.create_trained_policy(config, ckpt_dir)
        print(f"loaded pi05 policy from {ckpt_dir}")

        self.img_size = (224, 224)
        self.observation_window = None
        self.instruction = None

    def set_img_size(self, img_size):
        self.img_size = img_size

    def set_language(self, instruction):
        self.instruction = instruction
        print(f"instruction: {instruction}")

    def update_observation_window(self, img_arr, state):
        # img_arr is [head, front] -- see encode_obs in deploy_policy.py.
        img_head, img_front = img_arr[0], img_arr[1]

        # openpi expects [channel, height, width]; the sim hands us [h, w, c].
        to_chw = lambda im: np.transpose(im, (2, 0, 1))

        self.observation_window = {
            "state": state,
            "images": {
                "cam_high": to_chw(img_head),
                "cam_left_wrist": to_chw(img_front),
                # cam_right_wrist intentionally omitted -> AlohaInputs zero-fills it
                # and sets its image_mask False, matching training.
            },
            "prompt": self.instruction,
        }

    def get_action(self):
        assert self.observation_window is not None, "update observation_window first"
        return self.policy.infer(self.observation_window)["actions"]

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
