import argparse
from isaaclab.app import AppLauncher # type: ignore
 
# For parsing and launching app do not modify unless you need custom arguments
parser = argparse.ArgumentParser(description="Spawn a conveyor belt")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

#Imports
import isaacsim.core.utils.prims as prim_utils # type: ignore
import isaaclab.sim as sim_utils # type: ignore
import omni.usd # type: ignore
from pxr import UsdPhysics, PhysxSchema, Usd, Gf # type: ignore
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR # type: ignore
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG # type: ignore
import time
import torch
from isaaclab.assets import ( # type: ignore
    Articulation,
    ArticulationCfg,
    AssetBaseCfg,
    RigidObject,
    RigidObjectCfg,
    RigidObjectCollection,
    RigidObjectCollectionCfg,
    DeformableObject,
    DeformableObjectCfg
) 
import math
import random
from isaaclab.sim import SimulationContext # type: ignore

from isaaclab.utils import configclass # type: ignore
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg # type: ignore
import isaaclab.sim as sim_utils # type: ignore
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR,ISAAC_NUCLEUS_DIR # type: ignore
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg # type: ignore
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns # type: ignore


import math
import torch

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass



deltaT = 0.01
infeedVelocity = 0.0467
outfeedVelocity = 0.1333
infeed_y_offset = -0.5
outfeed_y_offset = 0.1
pancakes_per_container = 6
infeed_gen_dist = 0.095 
outfeed_gen_dist = 0.230
potential_y = [-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4]

INFEED_CONVEYOR_CFG = RigidObjectCfg(
    spawn=sim_utils.CuboidCfg(size=[4.0, 1, 0.9],
                              collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                              mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
                              rigid_props=sim_utils.RigidBodyPropertiesCfg(
                                  kinematic_enabled=True,
                              ),
                              ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(-2, infeed_y_offset, 0.45)),
)
OUTFEED_CONVEYOR_CFG = RigidObjectCfg(
    spawn=sim_utils.CuboidCfg(size=[4.0, 0.2, 0.8],
                              collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                              mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
                              rigid_props=sim_utils.RigidBodyPropertiesCfg(
                                  kinematic_enabled=True,
                              ),
                              ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(-2, outfeed_y_offset, 0.4)),
)





@configclass
class PancakeSceneCfg(InteractiveSceneCfg):
    """Configuration for scene."""
    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
 

    infeed_conveyor: RigidObjectCfg = INFEED_CONVEYOR_CFG.replace(prim_path="{ENV_REGEX_NS}/infeed_conveyor")
    outfeed_conveyor: RigidObjectCfg = OUTFEED_CONVEYOR_CFG.replace(prim_path="{ENV_REGEX_NS}/outfeed_conveyor")

    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot", init_state=ArticulationCfg.InitialStateCfg(pos=(-2.5, -0.2, 2.2), rot=[0, 1, 0, 0]))


@configclass
class ActionsCfg:
    joint_efforts = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["panda_joint.*"], scale=0.5)

@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
        # on reset
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )

@configclass
class CartpoleEnvCfg(ManagerBasedEnvCfg):
    """Configuration for the cartpole environment."""

    # Scene settings
    scene = PancakeSceneCfg(num_envs=2, env_spacing=2.5)
    # Basic settings
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # viewer settings
        self.viewer.eye = [4.5, 0.0, 6.0]
        self.viewer.lookat = [0.0, 0.0, 2.0]
        # step settings
        self.decimation = 4  # env step every 4 sim steps: 200Hz / 4 = 50Hz
        # simulation settings
        self.sim.dt = 0.005  # sim step every 5ms: 200Hz

def main():
    """Main function."""
    # parse the arguments
    env_cfg = CartpoleEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # setup base environment
    env = ManagerBasedEnv(cfg=env_cfg)

    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 300 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # sample random actions
            joint_efforts = torch.randn_like(env.action_manager.action)
            # step the environment
            obs, _ = env.step(joint_efforts)
            # print current orientation of pole
            print("[Env 0]: Pole joint: ", obs["policy"][0][1].item())
            # update counter
            count += 1

    # close the environment
    env.close()
