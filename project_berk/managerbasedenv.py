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





deltaT = 0.03
infeedVelocity = 0.0467
outfeedVelocity = 0.1333
infeed_y_offset = -0.5
outfeed_y_offset = 0.1+0.001
pancakes_per_container = 6
infeed_gen_dist = 0.095 
outfeed_gen_dist = 0.230
potential_y = [-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4] # idea make a map of potential y's and spawn them randomly

item_veloctiy = 10 # m/s

num_containers = math.ceil(4 / (outfeed_gen_dist+0.195)) 
num_pancake_row = math.ceil(4/(infeed_gen_dist+0.09))

device = "cuda:0"


pancake_cfg =  RigidObjectCfg(
        spawn=sim_utils.CylinderCfg(
                radius=0.045,
                height=0.010,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled= True,  
                                                             kinematic_enabled=False,
                                                             enable_gyroscopic_forces=False,
                                                             retain_accelerations=False,
                                                             solver_position_iteration_count=4, 
                                                             solver_velocity_iteration_count=0,
                                                             linear_damping=0.0,
                                                             angular_damping=0.0,
                                                             max_linear_velocity=1000.0,
                                                             max_angular_velocity=1000.0,
                                                             max_depenetration_velocity=1.0,
                                                             sleep_threshold=0.05,
                                                             stabilization_threshold=0.01,
                                                             ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
            ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )

container_cfg =  RigidObjectCfg(
        spawn=sim_utils.CuboidCfg(
                size=(0.195,0.125 , 0.005),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled= True,  
                                                             kinematic_enabled=False,
                                                             enable_gyroscopic_forces=False,
                                                             retain_accelerations=False,
                                                             solver_position_iteration_count=4, 
                                                             solver_velocity_iteration_count=0,
                                                             linear_damping=0.0,
                                                             angular_damping=0.0,
                                                             max_linear_velocity=1000.0,
                                                             max_angular_velocity=1000.0,
                                                             max_depenetration_velocity=1.0,
                                                             sleep_threshold=0.05,
                                                             stabilization_threshold=0.01,),
                mass_props=sim_utils.MassPropertiesCfg(mass=6.0),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0), metallic=0.2),
            ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )


pancake_cfg_dict = {}

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

 
def spawn_object(i):
    pancake_cfg_dict = {}
    #This is for spawning objects onto the conveyor.
 
    for index in range(i):
        spawn_location = [-3.5, infeed_y_offset - 2, index* 0.01]
        pancake = pancake_cfg.copy()
        # pancake.prim_path = "{ENV_REGEX_NS}/pancake_" + str(i+1) + "_" + str(index+1)

        pancake.init_state = RigidObjectCfg.InitialStateCfg(pos=spawn_location) 

        key = f'pancake_{index+1}'
        pancake_cfg_dict[key] = pancake.replace(prim_path="/World/envs/env_.*/"+key)

    return pancake_cfg_dict


def spawn_container(container_num):
    container_cfg_dict={}
    num = math.ceil(container_num)
    for i in range(num):
        spawn_location = [-3.5, outfeed_y_offset + 1, 0.005 * i]
        container = container_cfg.copy()
        container.init_state = RigidObjectCfg.InitialStateCfg(pos=spawn_location) 

        key = f'container_{i+1}'
        container_cfg_dict[key] = container.replace(prim_path="/World/envs/env_.*/"+key)

    return container_cfg_dict








combined_dic = {}
container_dic = {}
 
    # Spawn the object and get the dictionary
combined_dic = spawn_object(num_pancake_row* len(potential_y) )
total_pancakes = len(combined_dic.keys())
container_dic = spawn_container(num_containers)
total_containers = len(container_dic.keys())

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


    pancake_collection: RigidObjectCollectionCfg = RigidObjectCollectionCfg(rigid_objects=combined_dic)
    container_collection: RigidObjectCollectionCfg = RigidObjectCollectionCfg(rigid_objects=container_dic) 

    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot", init_state=ArticulationCfg.InitialStateCfg(pos=(-2.5, -0.2, 2.2), rot=[0, 1, 0, 0]))



def move_conveyor(i):
    stage = omni.usd.get_context().get_stage()
    infeed_conveyor_prim = stage.GetPrimAtPath(f"/World/envs/env_{i}/infeed_conveyor")
    if infeed_conveyor_prim.IsValid():
        velocity_attr = infeed_conveyor_prim.GetAttribute("physics:velocity")
        velocity_attr.Set((infeedVelocity,0,0)) #meters per second
        print("Velocity set!")
    else:
        print("Infeed conveyor or infeed conveyor velocity not found!")
    outfeed_conveyor_prim = stage.GetPrimAtPath(f"/World/envs/env_{i}/outfeed_conveyor")
    if outfeed_conveyor_prim.IsValid():
        velocity_attr = outfeed_conveyor_prim.GetAttribute("physics:velocity")
        velocity_attr.Set((outfeedVelocity,0,0)) #meters per second
        print("Velocity set!")
    else:
        print("Outfeed conveyor or outfeed conveyor velocity not found!")

import numpy as np

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    robot = scene["robot"]
    # move_conveyor()
    for i in range(scene.num_envs):
        move_conveyor(i)
 

    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    i = 0
    container_index = 0
    batch = len(potential_y) 
    pancake_offset_count = 0
    container_offset_count = 0
    pick_workarea_1_movement = torch.zeros(2, 1)
    # Simulate physics
    while simulation_app.is_running():
        # reset


        pick_workarea_1_movement[:, 0] -= sim_dt * item_veloctiy

        pancakes_xy_pos = scene['pancake_collection'].data.object_com_pos_w[:,:,:2]
        pancakes_xy_pos -= scene.env_origins[:, None, :2]
        
        pick_workarea_1 = (pancakes_xy_pos[:, :, 0] > -3) & (pancakes_xy_pos[:, :, 0] < -2) & \
                        (pancakes_xy_pos[:, :, 1] > -1) & (pancakes_xy_pos[:, :, 1] < 0) # 判断 x 坐标是否在 (-2, -1) 范围内，且 y 坐标在 (-1, 1) 范围内
        if pick_workarea_1.any(): 

            # 遍历第一个维度（0 和 1）
            for env_i in range(scene.num_envs):


                if pick_workarea_1_movement[env_i] > 0:
                    continue
                # 获取当前维度的 True 索引
                true_indices = torch.nonzero(pick_workarea_1[env_i])[:, 0]  # 取当前维度的索引

                if len(true_indices) > 0:
                    # 随机选择一个满足条件的索引
                    
                    random_index = true_indices[torch.randint(0, len(true_indices), (1,))].item()
                    print(f"env_{env_i},随机选择的索引:", random_index)
                    object_default_state = scene['pancake_collection'].data.default_object_state[env_i,random_index,:].clone()
                    object_state = scene['pancake_collection'].data.object_com_pos_w[env_i,random_index,:].clone()
 
                    distance = torch.norm(object_default_state[:3] - object_state, p=1)

                    print(distance)  # 输出: tensor(9.)
                    pick_workarea_1_movement[env_i] = distance
                    
                    object_default_state[:3] += scene.env_origins[env_i]
                    scene['pancake_collection'].reset()
                    scene['pancake_collection'].write_object_com_state_to_sim(object_default_state.unsqueeze(0).unsqueeze(0), \
                                                                            scene['pancake_collection']._ALL_ENV_INDICES[env_i].unsqueeze(0), \
                                                                            scene['pancake_collection']._ALL_OBJ_INDICES[random_index].unsqueeze(0))
                    scene['pancake_collection'].update(sim_dt)
                else:
                    print(f"env_{env_i},没有满足条件的点。")




 
            
        # cycle
        container_delta_count = count - container_offset_count
        
        if (outfeedVelocity * deltaT * container_delta_count >= outfeed_gen_dist * container_index) and (outfeedVelocity * deltaT * (container_delta_count-1) < outfeed_gen_dist * container_index):

            deltaX_container = outfeedVelocity * deltaT * container_delta_count - outfeed_gen_dist * container_index  # need to suppliment the distance during increasing the time step
            print(f"[INFO]: Spawn container when {deltaT * container_delta_count}..and {deltaX_container =}.")
 
            if container_index >= total_containers:
                container_index = 0 
                container_offset_count = count
                print("----------------------------------------")
                print("[INFO]: Resetting containers state...")
                # scene['pancake_collection'].reset()
                
            

            container_initial_status = [
                    -3.5 + deltaX_container, outfeed_y_offset, 0.8  , 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                ]
            container_initial_status_tensor = torch.zeros(scene.num_envs, 13, device=device)
            container_initial_status_tensor += torch.tensor(container_initial_status, device=device)
            container_initial_status_tensor[:,:3] += scene.env_origins


            scene['container_collection'].reset()
            scene['container_collection'].write_object_com_state_to_sim(container_initial_status_tensor.unsqueeze(1),None,scene['container_collection']._ALL_OBJ_INDICES[container_index  ].unsqueeze(0))
            scene['container_collection'].update(sim_dt)
            print("----------------------------------------")
            container_index +=1





        pancake_delta_count = count - pancake_offset_count
        if (infeedVelocity * deltaT * pancake_delta_count >= infeed_gen_dist * i / batch) and (infeedVelocity * deltaT * (pancake_delta_count -1 ) > infeed_gen_dist * i /batch ):

            
            deltaX = infeedVelocity * deltaT * pancake_delta_count - infeed_gen_dist * i / batch  # need to suppliment the distance during increasing the time step
            print(f"[INFO]: Spawn pancake when {deltaT * pancake_delta_count}..and {deltaX =}.")
 
            if i >= total_pancakes:

                i = 0
                pancake_offset_count = count 
                print("----------------------------------------")
                print("[INFO]: Resetting pancakes state...") 
                # scene['infeed_conveyor'].reset()


            pancakes_initial_status = torch.zeros(batch, 13, device=device)
            pancakes_initial_status[:, 0] = -3.5 + deltaX  # 广播机制
            pancakes_initial_status[:, 1] = torch.tensor(potential_y) + infeed_y_offset  # 直接加法
            pancakes_initial_status[:, 2] = 0.9 +0.005  # 广播机制
            pancakes_initial_status[:, 3] = 1.0  # 广播机制

            # 利用广播机制将 scene.env_origins 扩展到 [2, batch, 3]
            expanded_env_origins = scene.env_origins[:, None, :]  # 形状变为 [2, 1, 3]
            # 创建一个 [2, batch, 13] 的张量，前 3 列是 expanded_env_origins，其余为 0
            final_env_origins = torch.zeros(scene.num_envs, batch, 13, device=device)
            final_env_origins[:, :, :3] = expanded_env_origins  # 广播机制会自动扩展
            pancakes_initial_status_tensor =  final_env_origins + pancakes_initial_status

            object_start_index = i
            object_end_index = i + batch if i + batch <= total_pancakes - 1 else total_pancakes 
 

            scene['pancake_collection'].reset()
            scene['pancake_collection'].write_object_com_state_to_sim(pancakes_initial_status_tensor[:,object_start_index - i:object_end_index - i,:],None,scene['pancake_collection']._ALL_OBJ_INDICES[object_start_index:object_end_index] ) 
            scene['pancake_collection'].update(sim_dt)
            print("----------------------------------------")
            i += batch

 


        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        scene.update(sim_dt)
 




def main():

    """Main function."""
 

    # Load kit helper
    # sim_cfg = sim_utils.SimulationCfg(dt=0.005,device=args_cli.device)
    sim_cfg = sim_utils.SimulationCfg(dt=deltaT,device=device)

    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([6.5, 0.0, 8.0], [0.0, 0.0, 3.0])
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