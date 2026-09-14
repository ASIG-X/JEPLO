# This file is part of JEPLO: Joint-Embedding Predictive Learning for LiDAR-Based Legged Locomotion
#
# Copyright (c) 2026 Qihao Yuan
#
# Developer: Qihao Yuan <qihao.yuan@rug.nl>
#
# For commercial use, please contact me at <qihao.yuan@rug.nl> or Kailai Li at <kailai.li@liu.se>.
#
# This file is subject to the terms and conditions outlined in the 'LICENSE' file,
# which is included as part of this source code package.

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

import argparse
import traceback

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=3)
parser.add_argument("--student", action="store_true", help="Play the student branch instead of the teacher.")
parser.add_argument("--student_env_ratio", type=float, default=0.25, help="Fraction of CTS environments using student actions.")
parser.add_argument("--onnx_test_samples", type=int, default=0)
parser.add_argument("--onnx_test_sample_probability", type=float, default=0.05)
parser.add_argument("--onnx_test_seed", type=int, default=None)
parser.add_argument("--onnx_test_log_interval", type=int, default=25)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

if args_cli.onnx_test_samples > 0:
    args_cli.enable_cameras = True
    print(f"[INFO] ONNX test case export enabled: {args_cli.onnx_test_samples} samples per model.")
    print("[INFO] Cameras enabled for depth image capture.")

# set task name
task_name = "Unitree-Go2-Locomotion"
print("=" * 50)
print("Playing locomotion policy.")
print("=" * 50)

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Start playing after the app is launched."""

import os
import time

import go2_parkour.tasks  # noqa: F401
import gymnasium as gym
import torch
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from go2_parkour.rsl_rl_wrapper import RslRlVecEnvWrapper
from go2_parkour.tasks.go2_loco_env_window import Go2LocoEnvWindow
from rsl_rl.runners.on_policy_runner_teacher_cts import OnPolicyRunnerTeacherCTS


def main():
    if args_cli.onnx_test_samples > 0 and not args_cli.student:
        raise ValueError("--onnx_test_samples requires --student.")

    # parse configuration
    env_cfg = parse_env_cfg(
        task_name,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg = cli_args.parse_rsl_rl_cfg(task_name, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    print(f"resume path: {resume_path}")
    print(f"log dir: {log_dir}")

    # update env cfg
    env_cfg.is_play_env = True
    env_cfg.ui_window_class_type = Go2LocoEnvWindow
    env_cfg.events.randomize_push_robot = None

    # create the environment
    env = gym.make(task_name, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # load previously trained model
    train_cfg = agent_cfg.to_dict()
    train_cfg["algorithm"]["student_env_ratio"] = args_cli.student_env_ratio
    print("=" * 50)
    print("CTS checkpoint playback enabled.")
    print(f"  branch: {'student' if args_cli.student else 'teacher'}")
    print("=" * 50)

    ppo_runner = OnPolicyRunnerTeacherCTS(
        env,
        train_cfg,
        log_dir=None,
        device=agent_cfg.device,
    )
    ppo_runner.load(resume_path, load_optimizer=False)

    # export policy to onnx
    export_succeeded = False
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    if args_cli.student:
        print("\n\nExporting the CTS student policy and sensor estimator: [ONNX]")
        try:
            ppo_runner.export_policy_onnx(export_model_dir)
            export_succeeded = True
            print("*** successfully exported CTS student policy and sensor estimator to onnx. ***")
        except Exception:
            import traceback

            traceback.print_exc()
            print("*" * 50)
            print("failed to export CTS student onnx models.")
            print("*" * 50)
    else:
        print("[INFO] Skipping ONNX export. Export is CTS-student only.")
    if args_cli.onnx_test_samples > 0 and not export_succeeded:
        raise RuntimeError("ONNX export failed, so ONNX test case collection was not started.")

    # obtain the trained policy for inference
    if args_cli.student:
        if args_cli.onnx_test_samples > 0:
            test_case_dir = os.path.join(export_model_dir, "test_cases")
            print(f"[INFO] Writing ONNX test cases to: {test_case_dir}")
            policy = ppo_runner.get_inference_policy(
                device=env.unwrapped.device,
                onnx_test_case_dir=test_case_dir,
                onnx_test_case_samples=args_cli.onnx_test_samples,
                onnx_test_case_seed=args_cli.onnx_test_seed,
                onnx_test_case_sample_probability=args_cli.onnx_test_sample_probability,
                onnx_test_case_log_interval=args_cli.onnx_test_log_interval,
            )
        else:
            policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    else:
        policy = ppo_runner.get_teacher_inference_policy(device=env.unwrapped.device)

    obs, extras = env.get_observations()
    timestep = 0
    dones = None

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            if args_cli.student:
                actions = policy(obs, extras)
            else:
                actions = policy(obs)
            # env stepping
            obs, _, dones, infos = env.step(actions)
            if args_cli.student:
                extras = infos
                policy.reset(dones.bool())

        timestep += 1

        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        if args_cli.onnx_test_samples > 0 and policy.onnx_test_cases_complete:
            print(f"[INFO] Collected requested ONNX test cases after {timestep} play steps. Exiting.")
            break

    # close the simulator
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
