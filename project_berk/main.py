
import argparse
from isaaclab.app import AppLauncher
import numpy

# For parsing and launching app do not modify unless you need custom arguments
parser = argparse.ArgumentParser(description="Spawn a conveyor belt")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

#Imports
import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
import omni.usd
from pxr import UsdPhysics, PhysxSchema, Usd, Gf
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
# from isaaclab.assets import RigidObject, RigidObjectCfg
import time
import torch
from isaaclab.assets import (
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

import random
from isaaclab.sim import SimulationContext

from isaaclab.utils import configclass
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR,ISAAC_NUCLEUS_DIR
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
from isaaclab_assets import UR10_CFG
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG

deltaT = 0.01
infeedVelocity = 0.0467
outfeedVelocity = 0.1333
infeed_y_offset = -0.5
outfeed_y_offset = 0.1
pancakes_per_container = 6
infeed_gen_dist = 0.095 
outfeed_gen_dist = 0.230
potential_y = [-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4] # idea make a map of potential y's and spawn them randomly




pancake_cfg =  RigidObjectCfg(
        spawn=sim_utils.CylinderCfg(
                radius=0.045,
                height=0.010,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
            ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )

container_cfg =  RigidObjectCfg(
        spawn=sim_utils.CuboidCfg(
                size=(0.195,0.125 , 0.005),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False),
                mass_props=sim_utils.MassPropertiesCfg(mass=6.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0), metallic=0.2),
            ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )


pancake_cfg_dict = {}

INFEED_CONVEYOR_CFG = RigidObjectCfg(
    spawn=sim_utils.CuboidCfg(size=[8.0, 1, 0.9],
                              collision_props=sim_utils.CollisionPropertiesCfg(),
                              mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
                              rigid_props=sim_utils.RigidBodyPropertiesCfg(
                                  kinematic_enabled=True,
                              ),
                              ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0, infeed_y_offset, 0.45)),
)
OUTFEED_CONVEYOR_CFG = RigidObjectCfg(
    spawn=sim_utils.CuboidCfg(size=[8.0, 0.2, 0.8],
                              collision_props=sim_utils.CollisionPropertiesCfg(),
                              mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
                              rigid_props=sim_utils.RigidBodyPropertiesCfg(
                                  kinematic_enabled=True,
                              ),
                              ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0, outfeed_y_offset, 0.4)),
)
 

def spawn_object(i):
    pancake_cfg_dict = {}
    #This is for spawning objects onto the conveyor.
 
    for index in range(i):
        spawn_location = [-3.5, infeed_y_offset - 2, 0]
        pancake = pancake_cfg.copy()
        # pancake.prim_path = "{ENV_REGEX_NS}/pancake_" + str(i+1) + "_" + str(index+1)

        pancake.init_state = RigidObjectCfg.InitialStateCfg(pos=spawn_location) 

        key = f'pancake_{index+1}'
        pancake_cfg_dict[key] = pancake.replace(prim_path="/World/envs/env_.*/"+key)

    return pancake_cfg_dict
import math

def spawn_container(pancake_num):
    container_cfg_dict={}
    num = math.ceil(pancake_num / pancakes_per_container)
    for i in range(num):
        spawn_location = [-3.5, outfeed_y_offset + 1, 0]
        container = container_cfg.copy()
        container.init_state = RigidObjectCfg.InitialStateCfg(pos=spawn_location) 

        key = f'container_{i+1}'
        container_cfg_dict[key] = container.replace(prim_path="/World/envs/env_.*/"+key)

    return container_cfg_dict





combined_dic = {}
container_dic = {}
 
    # Spawn the object and get the dictionary
combined_dic = spawn_object(20 * len(potential_y) )
    # Combine it with the existing dictionary 
# spawn container based on the pancakes

total_pancakes = len(combined_dic.keys())

container_dic = spawn_container(total_pancakes)

@configclass
class PancakeSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""


    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
 

    infeed_conveyor: RigidObjectCfg = INFEED_CONVEYOR_CFG.replace(prim_path="{ENV_REGEX_NS}/infeed_conveyor")
    outfeed_conveyor: RigidObjectCfg = OUTFEED_CONVEYOR_CFG.replace(prim_path="{ENV_REGEX_NS}/outfeed_conveyor")

    # franka: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot", init_state=ArticulationCfg.InitialStateCfg(pos=(-2.5, -0.2, 2.2), rot=[0, 1, 0, 0]))

    robot: ArticulationCfg = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot", init_state=ArticulationCfg.InitialStateCfg(pos=(-2.5, -0.2, 2.5), rot=[0, 1, 0, 0]))

    pancake_collection: RigidObjectCollectionCfg = RigidObjectCollectionCfg(rigid_objects=combined_dic)
    container_collection: RigidObjectCollectionCfg = RigidObjectCollectionCfg(rigid_objects=container_dic)


def move_conveyor():
    stage = omni.usd.get_context().get_stage()
    infeed_conveyor_prim = stage.GetPrimAtPath("/World/envs/env_0/infeed_conveyor")
    if infeed_conveyor_prim.IsValid():
        velocity_attr = infeed_conveyor_prim.GetAttribute("physics:velocity")
        velocity_attr.Set((infeedVelocity,0,0)) #meters per second
        print("Velocity set!")
    else:
        print("Infeed conveyor or infeed conveyor velocity not found!")
    outfeed_conveyor_prim = stage.GetPrimAtPath("/World/envs/env_0/outfeed_conveyor")
    if outfeed_conveyor_prim.IsValid():
        velocity_attr = outfeed_conveyor_prim.GetAttribute("physics:velocity")
        velocity_attr.Set((outfeedVelocity,0,0)) #meters per second
        print("Velocity set!")
    else:
        print("Outfeed conveyor or outfeed conveyor velocity not found!")



def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""

    move_conveyor()
    # conveyor_status = scene['conveyor'].data.default_root_state.clone()
    # conveyor_status[:,:3] = conveyor_status[:,:3] - scene.env_origins
    # conveyor_status[:,7] = 10
    # scene['conveyor'].write_root_state_to_sim(conveyor_status)


    robot = scene["robot"]
    pancake_objects = scene['pancake_collection'].cfg.rigid_objects

    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    count2=0
    i = 0
    container_index = 0
    batch = len(potential_y)
    container_used_set = set() 
    pancake_used_set = set()

    #Task Variables
    # touching = False
    # target_pos = None
    # target_index = None
    # infeed_y_offset = 0.45

    # Create controller
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # Define goals for the arm
    ee_goals = [
        [1, 0, 0, 0, 0, 0, 1],
        [-1, 0, 0, 0, 0, 0, 1],
    ]
    ee_goals = torch.tensor(ee_goals, device=sim.device)
    # Track the given command
    current_goal_idx = 0
    # Create buffers to store actions
    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
    ik_commands[:] = ee_goals[current_goal_idx]
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["ee_link"])

    # Resolving the scene entities
    robot_entity_cfg.resolve(scene)
    # Obtain the frame index of the end-effector
    # For a fixed base robot, the frame index is one less than the body index. This is because
    # the root body is not included in the returned Jacobians.
    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]

    joint_pos_des = 0

    # Simulate physics
    while simulation_app.is_running():
        # reset

        if i >len(pancake_objects):
            # reset counters
            sim_time = 0.0
            count = 0
            i = 0
            container_index = 0
            container_used_set = set() 
            pancake_used_set = set()
            print("----------------------------------------")
            print("[INFO]: Resetting object state...")
            scene['pancake_collection'].reset()
            scene['infeed_conveyor'].reset()
    
 
        if (outfeedVelocity * deltaT * count >= outfeed_gen_dist * container_index) and (outfeedVelocity * deltaT * (count-1) < outfeed_gen_dist * container_index):
            containers_status = scene['container_collection'].data.object_state_w.clone() 
            deltaX_container = outfeedVelocity * deltaT * count - outfeed_gen_dist * container_index  # need to suppliment the distance during increasing the time step
            print(f"[INFO]: Spawn container when {deltaT * count}..and {deltaX_container =}.")
 

            containers_status[:, container_index, 0] = -3.5 +deltaX_container
            containers_status[:, container_index, 1] = outfeed_y_offset
            containers_status[:, container_index, 2] = 0.8 + 0.003
            containers_status[:, container_index, 3] = 1.0
            containers_status[:, container_index, 4] = 0.0
            containers_status[:, container_index, 5] = 0.0
            containers_status[:, container_index, 6] = 0.0
            containers_status[:, container_index, 7] = 0.0
            containers_status[:, container_index, 8] = 0.0
            containers_status[:, container_index, 9] = 0.0
            containers_status[:, container_index, 10] = 0.0
            containers_status[:, container_index, 11] = 0.0
            containers_status[:, container_index, 12] = 0.0

            containers_status[:, container_index, :3] = containers_status[:, container_index, :3] + scene.env_origins
            # container_index = math.floor( i / pancakes_per_container)

            scene.reset()
            scene['container_collection'].write_object_com_state_to_sim(containers_status[:, container_index, :].unsqueeze(1),None,scene['container_collection']._ALL_OBJ_INDICES[container_index].unsqueeze(0))
            
            print("----------------------------------------")
            container_index +=1





        
        if (infeedVelocity * deltaT * count >= infeed_gen_dist * i / batch) and (infeedVelocity * deltaT * (count -1 ) > infeed_gen_dist * i /batch ):

            pancakes_status = scene['pancake_collection'].data.object_state_w.clone() 
            indices = []
            deltaX = infeedVelocity * deltaT * count - infeed_gen_dist * i / batch  # need to suppliment the distance during increasing the time step
            print(f"[INFO]: Spawn pancake when {deltaT * count}..and {deltaX =}.")
 
            for index, key in enumerate(pancake_objects.keys()):
                if index >= i and index < i + batch:
                    pancakes_status[:,index,0] = -3.5 +  deltaX
                    pancakes_status[:,index,1] = potential_y[index - i] + infeed_y_offset
                    pancakes_status[:,index,2] = 0.9+0.005
                    pancakes_status[:,index,3] = 1.
                    pancakes_status[:,index,4] = 0.
                    pancakes_status[:,index,5] = 0.
                    pancakes_status[:,index,6] = 0.
                    pancakes_status[:,index,7] = 0.
                    pancakes_status[:,index,8] = 0.
                    pancakes_status[:,index,9] = 0.
                    pancakes_status[:,index,10] = 0.
                    pancakes_status[:,index,11] = 0.
                    pancakes_status[:,index,12] = 0.
                    indices.append(index)
                    pancakes_status[:,index,:3] = pancakes_status[:,index,:3] + scene.env_origins
            i += batch
            if len(indices) > 0:
                scene.reset()
                scene['pancake_collection'].write_object_com_state_to_sim(pancakes_status[:,indices,:],None,scene['pancake_collection']._ALL_OBJ_INDICES[indices] ) 
                print("----------------------------------------")
 
 
            if count2 % 150 == 0:
                # reset time
                count2 = 0
                # reset joint state
                joint_pos = robot.data.default_joint_pos.clone()
                joint_vel = robot.data.default_joint_vel.clone()
                robot.write_joint_state_to_sim(joint_pos, joint_vel)
                robot.reset()
                # reset actions
                ik_commands[:] = ee_goals[current_goal_idx]
                joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()
                # reset controller
                diff_ik_controller.reset()
                diff_ik_controller.set_command(ik_commands)
                # change goal
                current_goal_idx = (current_goal_idx + 1) % len(ee_goals)
            else:
                # obtain quantities from simulation
                jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
                ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
                root_pose_w = robot.data.root_state_w[:, 0:7]
                joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
                # compute frame in root frame
                ee_pos_b, ee_quat_b = subtract_frame_transforms(
                    root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
                )
                # compute the joint commands
                joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)


        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        scene.write_data_to_sim()



        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        count2 += 1
        # update buffers
        scene.update(sim_dt)
 




def main():

    """Main function."""
 

    # Load kit helper
    # sim_cfg = sim_utils.SimulationCfg(dt=0.005,device=args_cli.device)
    sim_cfg = sim_utils.SimulationCfg(dt=deltaT,device=args_cli.device)

    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 8.0], [0.0, 0.0, 3.0])
    # Design scene
    scene_cfg = PancakeSceneCfg(num_envs=2, env_spacing=10.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


 

if __name__ == "__main__":
    main()
    simulation_app.close()