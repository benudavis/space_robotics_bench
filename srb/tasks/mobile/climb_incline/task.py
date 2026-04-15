import math
from dataclasses import MISSING
from typing import Sequence, Tuple

from srb.assets.scenery.incline import WALL_HEIGHT
import torch

from srb._typing import StepReturn
from srb.assets.scenery import Incline
from srb.core.asset import AssetBaseCfg
from srb.core.env import GroundEnv, GroundEnvCfg, GroundEventCfg, GroundSceneCfg
from srb.core.manager import EventTermCfg, SceneEntityCfg
from srb.core.marker import VisualizationMarkers, VisualizationMarkersCfg
from srb.core.mdp import reset_root_state_uniform
from srb.core.sim import PreviewSurfaceCfg, CollisionPropertiesCfg, CuboidCfg, RigidBodyMaterialCfg
from srb.core.sim.spawners.particles import PyramidParticlesSpawnerCfg
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

LENGTH = 5.0
WIDTH = 1.0
WALL_HEIGHT = 0.1

PARTICLE_SIZE = 0.01
PARTICLES_RATIO = 0.8

# Compute dimensions dynamically based on constants
DIM_X = round(LENGTH / PARTICLE_SIZE)
DIM_Y = round(WIDTH / PARTICLE_SIZE)
DIM_Z = round(max(1, round(0.25 * DIM_X)) / 5.0)

@configclass
class SceneCfg(GroundSceneCfg):


    scenery: Incline = Incline()

    # 1. The Ghost Rig (Using the CuboidCfg you already have!)
    wall_rig: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/wall_rig",
        spawn=CuboidCfg(
            size=(0.001, 0.001, 0.001), # Microscopically small
            collision_props=None,       # NO collisions (ghost object)
            # Do not add physics_material or rigid_body_props!
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            rot=rpy_to_quat(0.0, -25.0, 0.0),  # Matches the floor's pitch perfectly!
        ),
    )

    # 2. Spawn the walls as CHILDREN of the incline
    wall_x_pos: AssetBaseCfg = AssetBaseCfg(
        # Notice the path change! Nesting it under /incline inherits the rotation
        prim_path="{ENV_REGEX_NS}/wall_rig/wall_x_pos", 
        spawn=CuboidCfg(
            size=(LENGTH, 0.05, WALL_HEIGHT),
            collision_props=CollisionPropertiesCfg(),
            physics_material=RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            # Because it is a child, this position is now LOCAL to the tilted floor!
            pos=(0.0, 0.5*WIDTH + 0.025, 0.05), 
        ),
    )

    wall_x_neg: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/wall_rig/wall_x_neg",  # Nested
        spawn=CuboidCfg(
            size=(LENGTH, 0.05, WALL_HEIGHT),
            collision_props=CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, -0.5*WIDTH - 0.025, 0.05),
        ),
    )

    wall_y_pos: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/wall_rig/wall_y_pos",
        spawn=CuboidCfg(
            size=(0.05, WIDTH, WALL_HEIGHT),  # Thin, 2m wide, 10cm high
            collision_props=CollisionPropertiesCfg(),
            physics_material=RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.5*LENGTH + 0.025, 0.0, 0.05),  # At +X edge, slightly above ground
        ),
    )

    wall_y_neg: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/wall_rig/wall_y_neg",
        spawn=CuboidCfg(
            size=(0.05, WIDTH, WALL_HEIGHT),  # Thin, 2m wide, 10cm high
            collision_props=CollisionPropertiesCfg(),
            physics_material=RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-0.5*LENGTH - 0.025, 0.0, 0.05),  # At -X edge, slightly above ground
        ),
    )


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
                "y": (-0, 0),
                "z": (-0.36, -0.36),  # Ground is at -1.5 * sin(-25) = -0.63. need to factor in robot height of .215
                "roll": (0.0, 0.0),
                "pitch": (0, 0),
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
    episode_length_s: float = 10.0
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

    ## Particles
    particles: bool = True
    particles_size: float = PARTICLE_SIZE
    particles_ratio: float = PARTICLES_RATIO

    def __post_init__(self):

        # Configure particles BEFORE parent initialization
        # This ensures the particle system is properly set up during scene creation
        super().__post_init__()
        
        # Now update particle spawn configuration after parent has created it
        if self.scene.particles is not None:
            # Position particles right above the slope center
            self.scene.particles.init_state.pos = (0.0, 0.0, 5*.43 + 0.1)  # 5m slope height with 25deg angle + small offset
            
            # Update spawn to use PyramidParticlesSpawnerCfg with 5x2m bounds
            # This creates natural layered sand distribution
            dim_x = round(LENGTH / self.particles_size)  
            dim_y = round(WIDTH / self.particles_size)  
            dim_z = round(max(1, round(0.25 * dim_x))/5.0)  # Pyramid height
            
            # Clone and update the spawn config
            from dataclasses import replace
            self.scene.particles.spawn = replace(
                self.scene.particles.spawn,
                dim_x=dim_x,
                dim_y=dim_y,
                dim_z=dim_z,
            )

        # self.sim.device = "cuda" if torch.cuda.is_available() else "cpu"


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
