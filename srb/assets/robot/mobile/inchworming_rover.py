from srb.core.action import (
    ActionGroup,
    JointPositionToLimitsActionCfg,
    JointVelocityActionCfg,
)
from srb.core.actuator import ImplicitActuatorCfg
from srb.core.asset import ArticulationCfg, Frame, Transform, WheeledRobot
from srb.core.sim import (
    ArticulationRootPropertiesCfg,
    CollisionPropertiesCfg,
    RigidBodyPropertiesCfg,
    UsdFileCfg,
)
from srb.utils.cfg import configclass
from srb.utils.math import deg_to_rad, rpy_to_quat
from srb.utils.path import SRB_ASSETS_DIR

@configclass
class InchwormingActionGroup(ActionGroup):
    legs: JointPositionToLimitsActionCfg = JointPositionToLimitsActionCfg(
        asset_name="robot", joint_names=[".*_leg_joint"], rescale_to_limits=True
    )
    wheels: JointVelocityActionCfg = JointVelocityActionCfg(
        asset_name="robot", joint_names=[".*_wheel_joint"], scale=10.0
    )

class InchwormingRover(WheeledRobot):
    ## Model
    asset_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/inchworming_rover",
        spawn=UsdFileCfg(
            usd_path=SRB_ASSETS_DIR.joinpath("custom_assets")
            .joinpath("inchworming_rover")
            .joinpath("inchworming_rover.usd")
            .as_posix(),
            activate_contact_sensors=True,
            collision_props=CollisionPropertiesCfg(
                contact_offset=0.02, rest_offset=0.005
            ),
            rigid_props=RigidBodyPropertiesCfg(
                max_linear_velocity=1.5,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
        ),
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*_leg_joint"],
                velocity_limit=10.0,
                effort_limit=10.0,
                stiffness=800.0,
                damping=40.0,
            ),
            "base_drive": ImplicitActuatorCfg(
                joint_names_expr=[".*_wheel_joint"],
                velocity_limit=10.0,
                effort_limit=10.0,
                stiffness=100.0,
                damping=4000.0,
            ),
        },
    )

    ## Actions
    actions: ActionGroup = InchwormingActionGroup()

    ## Frames
    frame_base: Frame = Frame(prim_relpath="body")
    frame_payload_mount: Frame = Frame(
        prim_relpath="body",
        offset=Transform(pos=(0.0, 0.0, 0.1)),
    )
    frame_manipulator_mount: Frame = Frame(
        prim_relpath="body",
        offset=Transform(pos=(0.1, 0.0, 0.1)),
    )
    frame_front_camera: Frame = Frame(
        prim_relpath="body",
        offset=Transform(pos=(0.2, 0.0, 0.1)),
    )
