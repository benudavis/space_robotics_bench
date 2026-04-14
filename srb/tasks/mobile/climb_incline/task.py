import math
from dataclasses import MISSING
from typing import Sequence, Tuple

import torch

from srb._typing import StepReturn
from srb.assets.scenery import Incline
from srb.core.env import GroundEnv, GroundEnvCfg, GroundEventCfg, GroundSceneCfg
from srb.core.manager import EventTermCfg, SceneEntityCfg
from srb.core.marker import VisualizationMarkers, VisualizationMarkersCfg
from srb.core.mdp import reset_root_state_uniform
from srb.core.sim import PreviewSurfaceCfg
from srb.core.sim.spawners.shapes.extras.cfg import PinnedArrowCfg
from srb.utils.cfg import configclass
from srb.utils.math import (
    deg_to_rad,
    matrix_from_quat,
    rpy_to_quat,
    subtract_frame_transforms,
)

##############
### Config ###
##############


@configclass
class SceneCfg(GroundSceneCfg):
    ## Scenery
    scenery: Incline = Incline()


@configclass
class EventCfg(GroundEventCfg):
    ## Reset robot at the base of the incline
    randomize_robot_state: EventTermCfg = EventTermCfg(
        func=reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (-1.5, -1.5),  # Near the bottom of the 5m slope
                "y": (-0.1, 0.1),
                "z": (-0.4, -0.4),  # Ground is at -1.5 * sin(-25) = -0.63. need to factor in robot height of .215
                "roll": (0.0, 0.0),
                "pitch": (deg_to_rad(-25.0), deg_to_rad(-25.0)),
                "yaw": (deg_to_rad(90), deg_to_rad(90)),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )


@configclass
class TaskCfg(GroundEnvCfg):
    scenery: Incline = Incline()

    ## Scene
    scene: SceneCfg = SceneCfg()
    stack: bool = True

    ## Events
    events: EventCfg = EventCfg()

    ## Time
    episode_length_s: float = 30.0
    is_finite_horizon: bool = False

    ## Target
    target_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/target",
        markers={
            "target": PinnedArrowCfg(
                pin_radius=0.05,
                pin_length=2.0,
                tail_radius=0.05,
                tail_length=0.2,
                head_radius=0.04,
                head_length=0.08,
                visual_material=PreviewSurfaceCfg(emissive_color=(0.2, 0.8, 0.2)),
            )
        },
    )

    def __post_init__(self):
        super().__post_init__()


############
### Task ###
############


class Task(GroundEnv):
    cfg: TaskCfg

    def __init__(self, cfg: TaskCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        ## Get scene assets
        self._target_marker: VisualizationMarkers = VisualizationMarkers(
            self.cfg.target_marker_cfg
        )

        ## Initialize buffers
        self._goal = torch.zeros(self.num_envs, 7, device=self.device)
        self._goal[:, 0:3] = self.scene.env_origins
        # Goal is at the top of the incline (for 5m slope, x~=2.2)
        _angle = deg_to_rad(-25.0)
        self._goal[:, 0] += 2.2 * math.cos(_angle)
        self._goal[:, 1] += 0.0
        self._goal[:, 2] += -2.2 * math.sin(_angle) + 0.1  # slightly above, sin(-25) is negative, so -sin is positive -> up
        self._goal[:, 3:7] = torch.tensor(
            rpy_to_quat(0.0, -25.0, 0.0), device=self.device
        )

    def step(self, action: torch.Tensor):
        # Allow 0.5s for the robot to settle onto the plane by zeroing actions.
        # agent_rate is 1/25.0 (0.04s), so 0.5s is roughly 12 steps.
        action = torch.where(
            (self.episode_length_buf < 12).unsqueeze(-1),
            torch.zeros_like(action),
            action
        )
        return super().step(action)

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)

    def extract_step_return(self) -> StepReturn:
        ## Visualize target
        self._target_marker.visualize(self._goal[:, 0:3], self._goal[:, 3:7])

        _robot_pose = self._robot.data.root_link_pose_w
        return _compute_step_return(
            ## Time
            episode_length=self.episode_length_buf,
            max_episode_length=self.max_episode_length,
            truncate_episodes=self.cfg.truncate_episodes,
            ## Actions
            act_current=self.action_manager.action,
            act_previous=self.action_manager.prev_action,
            ## States
            # Root
            tf_pos_robot=_robot_pose[:, 0:3],
            tf_quat_robot=_robot_pose[:, 3:7],
            # Transforms (world frame)
            tf_pos_target=self._goal[:, 0:3],
            tf_quat_target=self._goal[:, 3:7],
        )


@torch.jit.script
def _compute_step_return(
    *,
    ## Time
    episode_length: torch.Tensor,
    max_episode_length: int,
    truncate_episodes: bool,
    ## Actions
    act_current: torch.Tensor,
    act_previous: torch.Tensor,
    ## States
    # Root
    tf_pos_robot: torch.Tensor,
    tf_quat_robot: torch.Tensor,
    # Transforms (world frame)
    tf_pos_target: torch.Tensor,
    tf_quat_target: torch.Tensor,
) -> StepReturn:
    num_envs = episode_length.size(0)
    device = episode_length.device

    ############
    ## States ##
    ############
    ## Transforms (world frame)
    # Robot -> Target
    tf_pos_robot_to_target, _tf_quat_robot_to_target = subtract_frame_transforms(
        t01=tf_pos_robot, q01=tf_quat_robot, t02=tf_pos_target, q02=tf_quat_target
    )
    dist_robot_to_target = torch.norm(tf_pos_robot_to_target, dim=-1)

    #############
    ## Rewards ##
    #############
    # Penalty: Action rate
    WEIGHT_ACTION_RATE = -0.5
    _action_rate = torch.mean(torch.square(act_current - act_previous), dim=1)
    penalty_action_rate = WEIGHT_ACTION_RATE * _action_rate

    # Reward: Position tracking | Robot <--> Target
    WEIGHT_POSITION_TRACKING = 10.0
    reward_position_tracking = WEIGHT_POSITION_TRACKING / (1.0 + dist_robot_to_target)

    # Reward: Reached target
    reward_reached_target = torch.where(
        dist_robot_to_target < 0.1,
        torch.ones_like(dist_robot_to_target) * 100.0,
        torch.zeros_like(dist_robot_to_target),
    )

    ##################
    ## Terminations ##
    ##################
    # Termination if robot reaches target or falls off the slope
    termination = (dist_robot_to_target < 0.8) | (tf_pos_robot[:, 2] < -1.5)
    # Truncation
    truncation = (
        episode_length >= max_episode_length
        if truncate_episodes
        else torch.zeros(num_envs, dtype=torch.bool, device=device)
    )

    return StepReturn(
        {
            "state": {
                "tf_pos_robot_to_target": tf_pos_robot_to_target,
            },
        },
        {
            "penalty_action_rate": penalty_action_rate,
            "reward_position_tracking": reward_position_tracking,
            "reward_reached_target": reward_reached_target,
        },
        termination,
        truncation,
    )
