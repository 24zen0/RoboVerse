"""FootballShoot config in SkillBench in Skillblender"""

from __future__ import annotations

from typing import Callable

import torch

from metasim.cfg.objects import ArticulationObjCfg
from metasim.cfg.simulator_params import SimParamCfg
from metasim.cfg.tasks.base_task_cfg import BaseRLTaskCfg
from metasim.cfg.tasks.skillblender.base_humanoid_cfg import BaseHumanoidCfg, BaseHumanoidCfgPPO

# from metasim.constants import PhysicStateType
from metasim.types import EnvState
from metasim.utils import configclass


# define new reward function
def reward_arti_obj_dof(env_states: EnvState, robot_name: str, cfg: BaseRLTaskCfg):
    arti_obj_dof_diff = env_states.objects["cabinet"].joint_pos - 0.0
    arti_obj_dof_error = torch.mean(torch.abs(arti_obj_dof_diff), dim=1)
    return torch.exp(-4 * arti_obj_dof_error), arti_obj_dof_error


def reward_wrist_arti_obj_distance(env_states: EnvState, robot_name: str, cfg: BaseRLTaskCfg):
    wrist_pos = env_states.robots[robot_name].body_state[:, cfg.wrist_indices, :3]  # [num_envs, 2, 3], two hands
    arti_obj_pos = env_states.objects["cabinet"].root_state[:, :3]  # [num_envs, 3]
    wrist_arti_obj_diff = wrist_pos - arti_obj_pos.unsqueeze(1)  # [num_envs, 2, 3]
    wrist_arti_obj_diff = torch.flatten(wrist_arti_obj_diff, start_dim=1)  # [num_envs, 6]
    wrist_arti_obj_error = torch.mean(torch.abs(wrist_arti_obj_diff), dim=1)
    return torch.exp(-4 * wrist_arti_obj_error), wrist_arti_obj_error


@configclass
class TaskCabinetCfgPPO(BaseHumanoidCfgPPO):
    seed = 5

    @configclass
    class Policy(BaseHumanoidCfgPPO.Policy):
        """Network config class for PPO."""

        # HRL
        num_dofs = 19
        frame_stack = 1
        command_dim = 8
        # Expert skills
        skill_dict = {
            "h1_wrist_walking": {
                "experiment_name": "h1_wrist_walking",
                "load_run": "2025_0628_232507",
                "checkpoint": -1,
                "low_high": (-2, 2),
            },
            "h1_wrist_reaching": {
                "experiment_name": "h1_wrist_reaching",
                "load_run": "2025_0724_202749",
                "checkpoint": -1,
                "low_high": (-1, 1),
            },
        }

    @configclass
    class Runner(BaseHumanoidCfgPPO.Runner):
        wandb = True
        policy_class_name = "ActorCriticHierarchical"
        algorithm_class_name = "PPO"
        num_steps_per_env = 60  # per iteration
        max_iterations = 15001  # 3001  # number of policy updates

        # logging
        save_interval = 500
        experiment_name = "task_cabinet"

    policy = Policy()
    runner = Runner()


@configclass
class TaskCabinetCfg(BaseHumanoidCfg):
    """Cfg class for Skillbench:CabinetClose."""

    task_name = "task_cabinet"
    env_spacing = 5.0
    decimation = 10
    sim_params = SimParamCfg(
        dt=0.001,
        contact_offset=0.01,
        solver_type=1,
        substeps=1,
        num_position_iterations=4,
        max_depenetration_velocity=1.0,
        rest_offset=0.0,
        num_velocity_iterations=0,
        bounce_threshold_velocity=0.1,
        replace_cylinder_with_capsule=False,
        friction_offset_threshold=0.04,
        default_buffer_size_multiplier=5,
        num_threads=10,
    )

    ppo_cfg = TaskCabinetCfgPPO()

    command_ranges = BaseHumanoidCfg.CommandRanges(
        lin_vel_x=[-0, 0], lin_vel_y=[-0, 0], ang_vel_yaw=[-0, 0], heading=[-0, 0]
    )

    num_actions = 19
    frame_stack = 1
    c_frame_stack = 3
    command_dim = 8
    num_single_obs = 3 * num_actions + 6 + command_dim  # see `obs_buf = torch.cat(...)` for details
    num_observations = int(frame_stack * num_single_obs)
    single_num_privileged_obs = 3 * num_actions + 18 + 8
    num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)

    commands = BaseHumanoidCfg.CommandsConfig(num_commands=4, resampling_time=8.0)

    reward_functions: list[Callable] = [reward_arti_obj_dof, reward_wrist_arti_obj_distance]
    reward_weights: dict[str, float] = {
        "wrist_arti_obj_distance": 1,
        "arti_obj_dof": 5,
    }

    objects = [
        ArticulationObjCfg(
            name="cabinet",
            urdf_path="roboverse_data/assets/gapartnet/45159/mobility_annotation_gapartnet.urdf",
            fix_base_link=True,
            enabled_gravity=False,
        ),
    ]

    init_states = [
        {
            "objects": {
                "cabinet": {
                    "pos": torch.tensor([1.5, 0.0, 1.0]),
                    "rot": torch.tensor([
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                    ]),
                    "dof_pos": {
                        "joint_0": 1.0,
                        "joint_1": 1.0,
                    },
                },
            },
            "robots": {
                "h1_wrist": {
                    "pos": torch.tensor([0.0, 0.0, 1.0]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    "dof_pos": {
                        "left_hip_yaw": 0.0,
                        "left_hip_roll": 0.0,
                        "left_hip_pitch": -0.4,
                        "left_knee": 0.8,
                        "left_ankle": -0.4,
                        "right_hip_yaw": 0.0,
                        "right_hip_roll": 0.0,
                        "right_hip_pitch": -0.4,
                        "right_knee": 0.8,
                        "right_ankle": -0.4,
                        "torso": 0.0,
                        "left_shoulder_pitch": 0.0,
                        "left_shoulder_roll": 0.0,
                        "left_shoulder_yaw": 0.0,
                        "left_elbow": 0.0,
                        "right_shoulder_pitch": 0.0,
                        "right_shoulder_roll": 0.0,
                        "right_shoulder_yaw": 0.0,
                        "right_elbow": 0.0,
                    },
                },
            },
        }
    ]
