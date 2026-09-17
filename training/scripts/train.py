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
import os
import shlex
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

launch_command = shlex.join(getattr(sys, "orig_argv", [sys.executable, *sys.argv]))
launch_cwd = os.getcwd()

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=10000)
parser.add_argument("--gui", action="store_true")
parser.add_argument("--wandb_proj", type=str, default=None)
parser.add_argument(
    "--student_env_ratio", type=float, default=0.25, help="Fraction of environments using student actions."
)
parser.add_argument(
    "--warmup_iters",
    type=int,
    nargs=2,
    default=(0, 0),
    metavar=("TEACHER_ONLY_END", "FULL_CTS_START"),
    help="CTS rollout schedule: teacher-only end iteration and full-CTS start iteration.",
)
parser.add_argument("--student_encoder_learning_rate", type=float, default=2.0e-4)
parser.add_argument("--student_encoder_loss_coef", type=float, default=1.0)
parser.add_argument("--disable_student_encoder_loss", action="store_true")
parser.add_argument(
    "--onnx_test_samples",
    type=int,
    default=10,
    help="Number of current training samples per ONNX model to export for reference validation.",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# --gui takes precedence over --headless
args_cli.headless = not args_cli.gui

# set task name
task_name = "Unitree-Go2-Locomotion"
print("=" * 50)
print("Training locomotion teacher policy.")
print("=" * 50)

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""After the app is launched, start training."""

from datetime import datetime

import go2_parkour.tasks  # noqa: F401
import gymnasium as gym
import torch
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from go2_parkour.rsl_rl_wrapper import RslRlVecEnvWrapper
from rsl_rl.runners.on_policy_runner_teacher_cts import OnPolicyRunnerTeacherCTS

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
    # load configs from the registry
    env_cfg = parse_env_cfg(
        task_name,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=True,
    )
    agent_cfg = cli_args.parse_rsl_rl_cfg(task_name, args_cli)

    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)

    # set number of environments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set maximum iterations
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    env_cfg.seed = agent_cfg.seed

    # set device: default cuda:0
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if not 0.0 <= args_cli.student_env_ratio <= 1.0:
        raise ValueError("--student_env_ratio must be in [0, 1].")
    if not 0 <= args_cli.warmup_iters[0] <= args_cli.warmup_iters[1]:
        raise ValueError("--warmup_iters must satisfy 0 <= TEACHER_ONLY_END <= FULL_CTS_START.")
    if args_cli.onnx_test_samples < 1:
        raise ValueError("--onnx_test_samples must be positive.")
    student_encoder_loss_coef = 0.0 if args_cli.disable_student_encoder_loss else args_cli.student_encoder_loss_coef

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    # set run name
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = run_timestamp
    wandb_run_name = run_timestamp
    if agent_cfg.run_name:
        log_dir = agent_cfg.run_name
        wandb_run_name = f"{agent_cfg.run_name}_{run_timestamp}"

    print(f"Exact experiment name requested from command line: {log_dir}")
    log_dir = os.path.join(log_root_path, log_dir)

    # update env config
    env_cfg.is_play_env = False

    # create environment
    env = gym.make(task_name, cfg=env_cfg, render_mode=None)

    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print("=" * 50)
        print(f"Loading checkpoint from: {resume_path}")
        print("=" * 50)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    train_cfg = agent_cfg.to_dict()
    train_cfg["algorithm"]["student_env_ratio"] = args_cli.student_env_ratio
    train_cfg["algorithm"]["warmup_iters"] = args_cli.warmup_iters
    train_cfg["algorithm"]["student_encoder_learning_rate"] = args_cli.student_encoder_learning_rate
    train_cfg["algorithm"]["student_encoder_loss_coef"] = student_encoder_loss_coef
    train_cfg["onnx_test_samples"] = args_cli.onnx_test_samples
    train_cfg["launch_command"] = launch_command
    train_cfg["launch_cwd"] = launch_cwd
    train_cfg["wandb_run_name"] = wandb_run_name
    print("=" * 50)
    print("CTS training enabled.")
    print(f"  student_env_ratio: {args_cli.student_env_ratio}")
    print(f"  warmup_iters: {args_cli.warmup_iters}")
    print(f"  student_encoder_loss_coef: {student_encoder_loss_coef}")
    print("=" * 50)

    # create runner from rsl-rl
    runner = OnPolicyRunnerTeacherCTS(
        env,
        train_cfg,
        log_dir=log_dir,
        device=agent_cfg.device,
        wandb_project=args_cli.wandb_proj,
    )

    # write git state to logs
    runner.add_git_repo_to_log(__file__)

    # load the checkpoint
    if agent_cfg.resume:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), train_cfg)

    # Treat max_iterations as the total run target.  On resume the runner's
    # counter points at the first iteration that has not yet been completed.
    num_learning_iterations = max(agent_cfg.max_iterations - runner.current_learning_iteration, 0)
    if agent_cfg.resume:
        print(
            f"[INFO] Resume target: {runner.current_learning_iteration} iterations already completed; "
            f"running {num_learning_iterations} more to reach {agent_cfg.max_iterations}."
        )

    # run training
    if num_learning_iterations > 0:
        runner.learn(num_learning_iterations=num_learning_iterations)
    else:
        print(f"[INFO] Training target of {agent_cfg.max_iterations} iterations is already complete.")

    # close the simulator
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
