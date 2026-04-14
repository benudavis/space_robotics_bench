from srb.core.asset import AssetBaseCfg, Terrain
from srb.core.sim import CollisionPropertiesCfg, CuboidCfg, RigidBodyMaterialCfg
from srb.utils.math import rpy_to_quat


class Incline(Terrain):
    asset_cfg: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/incline",
        spawn=CuboidCfg(
            size=(5.0, 3.0, 0.05),  # 5m long, 5m wide, 5cm thick
            collision_props=CollisionPropertiesCfg(),
            physics_material=RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            rot=rpy_to_quat(0.0, -25.0, 0.0),  # -25-degree pitch makes +X point UP
        ),
    )
