"""This is a training script for skillblender framework"""

from __future__ import annotations

import os
import shutil

from loguru import logger as log

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import rootutils
import torch

rootutils.setup_root(__file__, pythonpath=True)

import wandb

from metasim.cfg.scenario import ScenarioCfg
from roboverse_learn.rl.rsl_rl.rsl_rl.runners.on_policy_runner import OnPolicyRunner
from roboverse_learn.skillblender_rl.utils import get_args, get_load_path_safe, get_log_dir, get_wrapper


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    scenario = ScenarioCfg(
        task=args.task,
        robots=[args.robot],
        num_envs=args.num_envs,
        sim=args.sim,
        headless=args.headless,
        cameras=[],
    )

    use_wandb = args.use_wandb
    if use_wandb:
        wandb.init(project=args.wandb, name=args.run_name, entity=args.entity)

    log_dir = get_log_dir(args, scenario)
    load_path = get_load_path_safe(args, scenario)

    # Only check for model existence if we're actually trying to load one
    if load_path is not None and not os.path.exists(load_path):
        log.error(f"Model file {load_path} does not exist!")
        return

    task_wrapper = get_wrapper(args.task)
    env = task_wrapper(scenario)

    # dump snapshot of training config
    task_path = f"metasim/cfg/tasks/skillblender/{scenario.task.task_name}_cfg.py"
    if not os.path.exists(task_path):
        log.error(f"Task path {task_path} does not exist, please check your task name in config carefully")
        return

    shutil.copy2(task_path, log_dir)

    # Only copy model file if we're loading one
    if load_path is not None:
        shutil.copy2(load_path, log_dir)

    ppo_runner = OnPolicyRunner(
        env=env,
        train_cfg=env.train_cfg,
        device=device,
        log_dir=log_dir,
        wandb=use_wandb,
        args=args,
        load_path=load_path,
    )

    # Only load checkpoint if we have a load path
    if load_path is not None:
        ppo_runner.load(load_path)
        assert ppo_runner.current_learning_iteration > 0, (
            f"Checkpoint not loaded correctly! Current iteration: {ppo_runner.current_learning_iteration}"
        )

        # keep training after loading checkpoint
        remaining_iters = args.learning_iterations - ppo_runner.current_learning_iteration
        assert remaining_iters > 0, (
            f"Checkpoint iteration ({ppo_runner.current_learning_iteration}) >= max ({args.learning_iterations})"
        )
        ppo_runner.learn(num_learning_iterations=remaining_iters)
    else:
        # Start training from scratch
        ppo_runner.learn(num_learning_iterations=args.learning_iterations)


if __name__ == "__main__":
    args = get_args()
    train(args)
