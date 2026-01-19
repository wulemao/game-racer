#!/usr/bin/env python3

import os
import sys
import traceback
import yaml
import numpy as np
import torch
import jax
import jax.numpy as jnp
import pickle
import argparse
import rospy
import time
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PolygonStamped, Point32
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Float64, Int8
from visualization_msgs.msg import MarkerArray
from tf.transformations import quaternion_from_euler, euler_matrix
from shift_local_traj import ShiftLocalTraj

# Path Setup
_THIS_DIR = os.path.dirname(os.path.realpath(__file__))
_GAME_RACER_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _GAME_RACER_ROOT not in sys.path:
    sys.path.insert(0, _GAME_RACER_ROOT)
SIM_PARAMS_PATH = os.path.join(
    _GAME_RACER_ROOT,
    "sim_params",
)
from car_dynamics.models_jax import DynamicBicycleModel, DynamicParams
from car_dynamics.envs.car_env import OffroadCar
from car_dynamics.controllers_jax import WaypointGenerator
# from model_arch import SimpleModel

# --- Global Directory Definitions ---
parent_dir = os.path.dirname(_THIS_DIR)
data_dir = os.path.join(parent_dir, 'data')
os.makedirs(data_dir, exist_ok=True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)

class CarFleet:
    def __init__(self, config):
        self.config = config

        self.traj_helper = ShiftLocalTraj(config)
        # Determine number of cars from config
        self.num_cars = config.get('sim', {}).get('num_cars', 2)
        
        # [Fix 1] VIS flag from config (default to True if missing)
        self.vis_enabled = config.get('sim', {}).get('vis', True)

        self.cars = {}
        self.publishers = {}
        self.ep_no = 0
        self.buffer = []
        self.dataset = []
        
        # Load params from config
        dyn = config['car_dynamics']
        self.DT = dyn['DT']
        self.DT_torch = dyn['DT_torch']
        self.DELAY = dyn['DELAY']
        self.H = dyn['H']
        self.LF = dyn['LF']
        self.LR = dyn['LR']
        self.L_base = self.LF + self.LR
        
        # Simulation settings
        sim_config = config['sim'][config['sim']['type']]
        self.trajectory_type = sim_config['trajectory_type']
        self.trajectory_type = os.path.join(SIM_PARAMS_PATH, self.trajectory_type)
        
        self.init_car_params()
        self.init_env()
        self.init_publishers()

    def init_publishers(self):
        if not self.vis_enabled: return
        
        for i in range(self.num_cars):
            name_idx = f"_{i}" if i > 0 else ""
            # Create a dict for this car's publishers
            pubs = {
                'path_nn': rospy.Publisher(f'path_nn{name_idx}', Path, queue_size=1),
                'pose': rospy.Publisher(f'pose{name_idx}', PoseWithCovarianceStamped, queue_size=1),
                'odom': rospy.Publisher(f'odom{name_idx}', Odometry, queue_size=1),
                'body': rospy.Publisher(f'body{name_idx}', PolygonStamped, queue_size=1),
                'ref_traj': rospy.Publisher(f'ref_trajectory{name_idx}', Path, queue_size=1),
                'throttle': rospy.Publisher(f'throttle{name_idx}', Float64, queue_size=1),
                'steer': rospy.Publisher(f'steer{name_idx}', Float64, queue_size=1),
            }
            
            # Car 0 acts as the "Ego" for map visualization usually
            if i == 0:
                pubs['waypoint_list'] = rospy.Publisher('waypoint_list', Path, queue_size=1)
                pubs['left_boundary'] = rospy.Publisher('left_boundary', Path, queue_size=1)
                pubs['right_boundary'] = rospy.Publisher('right_boundary', Path, queue_size=1)
                pubs['raceline'] = rospy.Publisher('raceline', Path, queue_size=1)
            
            self.publishers[i] = pubs
        
        self.status_pub = rospy.Publisher('status', Int8, queue_size=1)

    def init_car_params(self):
        for i in range(self.num_cars):
            self.cars[i] = {
                'index': i,
                'curr_speed_factor': 1.0,
                'curr_lookahead_factor': 0.24 if i == 0 else 0.15,
                'curr_sf1': 0.2 if i == 0 else 0.1,
                'curr_sf2': 0.2 if i == 0 else 0.5,
                'blocking': 0.2,
                'last_i': -1,
                'curr_steer': 0.0,
                's': 0.0, 'e': 0.0, 'vx': 0.0, 'px': 0.0, 'py': 0.0, 'psi': 0.0, 'omega': 0.0,
                # Initialize obs with zeros to prevent crash on first tick
                'obs': np.zeros(6), 
                'action': np.array([0.0, 0.0]),
                'target_pos_tensor': [],
                'theta_diff': 0.0,
                # Add storage for curvature data
                'curv': 0.0,
                'curv_lookahead': 0.0
            }

    def init_env(self):
        # [Suggestion] Move start poses to YAML config if possible for flexibility
        # self.start_poses = self.config['sim'].get('start_poses', [
        #     [3., 5., -np.pi/2.-0.72],
        #     [0., 0., -np.pi/2.-0.5],
        #     [-2., -6., -np.pi/2.-0.5]
        # ])
        self.start_poses = self.config['sim'].get('start_poses', [
            [3., 5., -np.pi/2.-0.72],
            [0., 0., -np.pi/2.-0.5]
        ])
        
        # Dynamically extend start poses if we have more cars than poses
        while self.num_cars > len(self.start_poses):
            base = self.start_poses[len(self.start_poses) % 3]
            # Simple offset, caution: verify map bounds!
            self.start_poses.append([base[0], base[1]-10.0, base[2]])

        for i in range(self.num_cars):
            self.cars[i]['model_params'] = DynamicParams(num_envs=1, DT=self.DT, Sa=0.34, Sb=-0., Ta=20., Tb=.0, mu=0.5, delay=self.DELAY)
            self.cars[i]['dynamic_single'] = DynamicBicycleModel(self.cars[i]['model_params'])
            self.cars[i]['dynamic_single'].reset()
            
            h_factor = 2. if i == 0 else 1.
            self.cars[i]['waypoint_generator'] = WaypointGenerator(self.trajectory_type, self.DT, self.H, h_factor)
            
            self.cars[i]['env'] = OffroadCar({}, self.cars[i]['dynamic_single'])

    # def reset_episode(self):
    #     self.ep_no += 1
        
    #     # Determine the number of available starting poses
    #     n_poses = len(self.start_poses)
        
    #     # =========================================================================
    #     # [Strategy Generation]
    #     # Instead of hardcoding assignments for 3 cars, we generate a list of 
    #     # position indices for ALL cars dynamically.
    #     # =========================================================================
        
    #     # 1. Create a list of all available position indices: [0, 1, 2, ... N-1]
    #     available_indices = np.arange(n_poses)
        
    #     # 2. Choose Strategy (Uncomment the one you want to use)
        
    #     # --- Strategy A: Random Permutation (Replaces Original Logic) ---
    #     # This randomly shuffles the positions. 
    #     # Works for ANY number of cars. No bias.
    #     np.random.shuffle(available_indices)
        
    #     # --- Strategy B: Deterministic Round-Robin (Optimized Option 2) ---
    #     # Cyclic rotation based on episode number. 
    #     # Ensures even data coverage over time.
    #     # available_indices = np.array([(self.ep_no + i) % n_poses for i in range(n_poses)])

    #     for i in range(self.num_cars):
    #         # Get the assigned position index for car 'i'
    #         # We take the i-th element from the shuffled/generated list
    #         # Use modulo just in case num_cars > n_poses (safety fallback)
    #         pose_idx = available_indices[i % n_poses]
            
    #         pose = self.start_poses[pose_idx]
            
    #         # Reset environment for car i
    #         self.cars[i]['obs'] = self.cars[i]['env'].reset(pose=pose)
    #         self.cars[i]['last_i'] = -1
    #         self.cars[i]['curr_steer'] = 0.0
    #         self.cars[i]['waypoint_generator'].last_i = -1
            
    #         # Randomize parameters (cyclic)
    #         # This logic is already scalable (depends on self.num_cars)
    #         # It updates the parameters of one car every episode, rotating through the fleet.
    #         if self.ep_no % self.num_cars == i:
    #             self.cars[i]['curr_sf1'] = np.random.uniform(0.1, 0.5)
    #             self.cars[i]['curr_sf2'] = np.random.uniform(0.1, 0.5)
    #             self.cars[i]['curr_lookahead_factor'] = np.random.uniform(0.12, 0.5)
    #             self.cars[i]['curr_speed_factor'] = np.random.uniform(0.85, 1.1)
    #             self.cars[i]['blocking'] = np.random.uniform(0., 1.0)

    def reset_episode(self):
        """
        Resets the environment for a new episode. 
        Ego car (Car 0) is placed randomly, while opponents (Car 1, 2...) 
        maintain a fixed relative order where larger indices are placed further forward.
        """
        self.ep_no += 1
        
        # 0 is front, 1 is middle, 2 is back (indices of start_poses)
        n_poses = len(self.start_poses)
        available_indices = list(range(n_poses))
        
        # --- Step 1: Randomly select a position for the Ego car (Car 0) ---
        ego_pos_idx = np.random.choice(available_indices)
        
        # --- Step 2: Prepare slots for Opponents ---
        # Get remaining indices after Ego is placed
        remaining_slots = [idx for idx in available_indices if idx != ego_pos_idx]
        
        # To satisfy "larger index = further forward":
        # Therefore, we sort opponent_slots in descending order (back to front)
        # E.g., if remaining are [0, 2], sorted becomes [2, 0]
        opponent_slots = sorted(remaining_slots, reverse=True)
        
        for i in range(self.num_cars):
            # Determine which pose index to use for the current car
            if i == 0:
                # Ego car gets the pre-selected random position
                pose_idx = ego_pos_idx
            else:
                # Opponent cars take slots sequentially from the reversed list
                slot_index = (i - 1) % len(opponent_slots)
                pose_idx = opponent_slots[slot_index]
            
            # Get actual pose coordinates from the configuration
            pose = self.start_poses[pose_idx]
            
            # --- Step 3: Reset Car State and Environment ---
            self.cars[i]['obs'] = self.cars[i]['env'].reset(pose=pose)
            self.cars[i]['last_i'] = -1
            self.cars[i]['curr_steer'] = 0.0
            self.cars[i]['waypoint_generator'].last_i = -1
            
            # --- Step 4: Randomize Parameters (Cyclic Randomization) ---
            # Only one car updates its hyper-parameters per episode for controlled data collection
            if self.ep_no % self.num_cars == i:
                self.cars[i]['curr_sf1'] = np.random.uniform(0.1, 0.5)
                self.cars[i]['curr_sf2'] = np.random.uniform(0.1, 0.5)
                self.cars[i]['curr_lookahead_factor'] = np.random.uniform(0.12, 0.5)
                self.cars[i]['curr_speed_factor'] = np.random.uniform(0.85, 1.1)
                self.cars[i]['blocking'] = np.random.uniform(0., 1.0)
                
                # Log the new parameters for the specific car being randomized
                print(f"Episode {self.ep_no}: Randomized Car {i} params -> "
                    f"sf1: {self.cars[i]['curr_sf1']:.2f}, "
                    f"blocking: {self.cars[i]['blocking']:.2f}")

    def get_car_state_for_control(self, car_idx):
        c = self.cars[car_idx]
        return (c['s'], c['e'], c['vx'])

    def get_obs_state(self, idx):
        return self.cars[idx]['env'].obs_state()

    def append_to_buffer(self, state_vector):
        self.buffer.append(state_vector)

    def save_episode_data(self, file_path):
        if len(self.buffer) > 0:
            self.dataset.append(np.array(self.buffer))
            self.buffer = []
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(np.array(self.dataset), f)
            print(f"Dataset saved to {file_path}. Total episodes: {len(self.dataset)}")
        except Exception as e:
            print(f"Failed to save dataset: {e}")

    def slow_timer_callback(self, event):
        # Check flag inside instance
        if not self.vis_enabled: return
        self.visualize_all(rospy.Time.now())

    def visualize_all(self, now):
        if not self.vis_enabled: return

        # 1. Map Elements (from car 0)
        wg = self.cars[0]['waypoint_generator']
        def pub_path(pub, points):
            path = Path()
            path.header.frame_id = 'map'
            path.header.stamp = now
            for i in range(len(points)):
                pose = PoseStamped()
                pose.header.frame_id = 'map'
                pose.pose.position.x = float(points[i][0])
                pose.pose.position.y = float(points[i][1])
                path.poses.append(pose)
            pub.publish(path)

        if self.ep_no == 1 or True:
             pubs0 = self.publishers[0]
             if 'raceline' in pubs0: pub_path(pubs0['raceline'], wg.raceline)
             if 'left_boundary' in pubs0: pub_path(pubs0['left_boundary'], wg.left_boundary)
             if 'right_boundary' in pubs0: pub_path(pubs0['right_boundary'], wg.right_boundary)
             if 'waypoint_list' in pubs0: pub_path(pubs0['waypoint_list'], wg.waypoint_list_np)

        # 2. Cars
        for k in range(self.num_cars):
            c = self.cars[k]
            pubs = self.publishers[k]
            
            # Pose
            q = quaternion_from_euler(0, 0, c['psi'])
            pose = PoseWithCovarianceStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = now
            pose.pose.pose.position.x = c['px']
            pose.pose.pose.position.y = c['py']
            pose.pose.pose.orientation.w = q[3]
            pose.pose.pose.orientation.x = q[0]
            pose.pose.pose.orientation.y = q[1]
            pose.pose.pose.orientation.z = q[2]
            pubs['pose'].publish(pose)
            
            # Odom
            odom = Odometry()
            odom.header.frame_id = 'map'
            odom.header.stamp = now
            odom.pose.pose = pose.pose.pose
            odom.twist.twist.linear.x = c['vx']
            odom.twist.twist.linear.y = c['vy']
            odom.twist.twist.angular.z = c['omega']
            pubs['odom'].publish(odom)
            
            # Body
            pts = np.array([[self.LF, self.L_base/3], [self.LF, -self.L_base/3], [-self.LR, -self.L_base/3], [-self.LR, self.L_base/3]])
            R = euler_matrix(0, 0, c['psi'])[:2, :2]
            pts_w = np.dot(R, pts.T).T + np.array([c['px'], c['py']])
            body = PolygonStamped()
            body.header.frame_id = 'map'
            body.header.stamp = now
            for pt in pts_w:
                p = Point32()
                p.x, p.y = float(pt[0]), float(pt[1])
                body.polygon.points.append(p)
            pubs['body'].publish(body)
            
            # Ref Traj
            if len(c['target_pos_tensor']) > 0:
                traj_pts = np.array(c['target_pos_tensor'])
                pub_path(pubs['ref_traj'], traj_pts)

            # Controls
            pubs['throttle'].publish(Float64(c['action'][0]))
            pubs['steer'].publish(Float64(c['action'][1]))


class CarNode:
    def __init__(self, config_path, exp_name):
        self.config = self.load_config(config_path)
        
        # Use the experiment name provided via CLI
        self.exp_name = exp_name

        # Append suffix if MPC is enabled
        if self.config.get('mpc', {}).get('enabled', False):
            self.exp_name += '_mpc'
            
        self.fleet = CarFleet(self.config)
        
        self.i = 0
        self.ep_len = self.config['car_dynamics']['EP_LEN']
        self.path_save_path = os.path.join(data_dir, f'{self.exp_name}.pkl')
        
        # Ensure we reset at least once before starting timer to prevent 'obs' crash
        self.fleet.reset_episode() 
        
        # Main Timer
        self.timer = rospy.Timer(rospy.Duration(self.config['car_dynamics']['DT']), self.timer_callback)
        
        # [Fix 1] Only create slow timer if VIS is enabled
        if self.fleet.vis_enabled:
            self.slow_timer_ = rospy.Timer(rospy.Duration(10.0), self.fleet.slow_timer_callback)

    @staticmethod
    def load_config(path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def timer_callback(self, event):
        try:
            self.i += 1
            # Check Episode End
            if self.i > self.ep_len:
                print("--- Episode Done ---")
                for k in range(self.fleet.num_cars):
                    print(f"Car {k} progress: {self.fleet.cars[k]['s']:.2f}")
                
                self.fleet.save_episode_data(self.path_save_path)
                
                if self.fleet.ep_no > 242:
                    print("Max episodes reached. Exiting...")
                    rospy.signal_shutdown("Max episodes reached")
                    sys.exit(0)

                self.fleet.reset_episode()
                self.i = 1

            # --- 1. Trajectory Generation (Physics Step 1) ---
            for k in range(self.fleet.num_cars):
                c = self.fleet.cars[k]
                obs_tensor = jnp.array(c['obs'][:5])
                target_pos, _, s, e = c['waypoint_generator'].generate(
                    obs_tensor, dt=self.fleet.DT_torch, mu_factor=1.0
                )
                c['s'], c['e'] = s, e
                c['target_pos_tensor'] = target_pos
                c['curv'] = float(target_pos[0, 3])
                c['curv_lookahead'] = float(target_pos[-1, 3])
                c['theta'] = target_pos[0, 2]
                
                full = self.fleet.get_obs_state(k).tolist()
                c['px'], c['py'], c['psi'], c['vx'], c['vy'], c['omega'] = full
                c['theta_diff'] = np.arctan2(np.sin(c['theta']-c['psi']), np.cos(c['theta']-c['psi']))

            # --- 2. Control Loop ---
            mpc_enabled = self.config['mpc']['enabled']
            
            for k in range(self.fleet.num_cars):
                c = self.fleet.cars[k]
                opponents = [self.fleet.get_car_state_for_control(ok) for ok in range(self.fleet.num_cars) if ok != k]
                
                state_pose = (c['px'], c['py'], c['psi'])
                state_track = (c['s'], c['e'], c['vx'])
                
                kwargs = {
                    'sf1': c['curr_sf1'],
                    'sf2': c['curr_sf2'],
                    'lookahead_factor': c['curr_lookahead_factor'],
                    'v_factor': c['curr_speed_factor'],
                    'blocking_factor': c['blocking'],
                    'last_i': c['last_i']
                }

                wg = self.fleet.cars[0]['waypoint_generator']

                if mpc_enabled:
                    kwargs['lookahead_factor'] *= 2
                    kwargs['v_factor'] **= 2

                    steer, throttle, curv, curv_lookahead, c['last_i'] = self.fleet.traj_helper.mpc(
                        state_pose, state_track, wg, *opponents, **kwargs
                    )
                else:

                    steer, throttle, curv, curv_lookahead, c['last_i'] = self.fleet.traj_helper.pure_pursuit(
                        state_pose, state_track, wg, *opponents, **kwargs
                    )
                
                # Store the calculated curvatures!
                # c['curv'] = curv
                # c['curv_lookahead'] = curv_lookahead

                # Heuristic Corrections
                if abs(c['e']) > 0.55:
                    decay = np.exp(-3 * (abs(c['e']) - 0.55))
                    c['env'].state.vx *= decay
                    c['env'].state.vy *= decay
                    c['env'].state.psi += (1 - np.exp(-(abs(c['e']) - 0.55))) * c['theta_diff']
                    steer += (-np.sign(c['e']) - steer) * (1 - decay)

                if abs(c['theta_diff']) > 1.0:
                    throttle += 0.2

                c['action'] = np.array([throttle, steer])
                c['obs'], _, _, _ = c['env'].step(c['action'])

            # --- 3. Collision Physics ---
            self.fleet.calculate_collisions()

            # --- 4. Data Collection ---
            # Construct flattened state vector
            state_vector = []
            
            for k in range(self.fleet.num_cars): state_vector.append(self.fleet.cars[k]['s'])
            for k in range(self.fleet.num_cars): state_vector.append(self.fleet.cars[k]['e'])
            for k in range(self.fleet.num_cars):
                c = self.fleet.cars[k]
                # theta_diff
                state_vector.append(c['theta_diff'])
                
                # (vx, vy, omega)
                obs = c['obs']
                if len(obs) >= 6: 
                    state_vector.extend(obs[3:6])
                else: 
                    state_vector.extend([0., 0., 0.])
            
            # Now we can correctly append the real curvatures
            for k in range(self.fleet.num_cars): state_vector.append(self.fleet.cars[k]['curv'])
            for k in range(self.fleet.num_cars): state_vector.append(self.fleet.cars[k]['curv_lookahead'])
            
            # Hyperparameters
            for k in range(self.fleet.num_cars):
                c = self.fleet.cars[k]
                state_vector.append(c['curr_sf1'])
                state_vector.append(c['curr_sf2'])
                state_vector.append(c['curr_lookahead_factor'])
                state_vector.append(c['curr_speed_factor'])
                state_vector.append(c['blocking'])

            self.fleet.append_to_buffer(np.array(state_vector))

            # --- 5. Visualization ---
            if self.fleet.vis_enabled: self.fleet.visualize_all(rospy.Time.now())

        except Exception:
            traceback.print_exc()

# ==============================================================================
# 5. Main Entry Point
# ==============================================================================
def main(config_path, exp_name):
    """
    Main function combining robust error handling with ROS execution.
    """
    start_time = time.time()
    car_node = None

    try:
        # Initialize ROS node here with anonymous=True
        print(f"[INFO] Initializing ROS Node...")
        rospy.init_node('car_node', anonymous=True)

        print(f"[INFO] Initializing CarNode with config: {config_path}")
        print(f"[INFO] Experiment Name: {exp_name}")
        
        car_node = CarNode(config_path, exp_name)
        
        # Use rospy.spin() to keep the node alive and processing callbacks
        rospy.spin()

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt detected. Shutting down...")

    except rospy.ROSInterruptException:
        pass

    except SystemExit:
        pass

    except Exception:
        print("\n[ERROR] An unexpected error occurred:")
        traceback.print_exc()

    finally:
        total_time = time.time() - start_time
        print(f"\n[EXIT] Total wall time: {total_time / 60.0:.2f} minutes")
        
        if car_node:
            del car_node
            
        sys.stdout.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    default_yaml = 'collect_args.yaml'
    
    # [Added] Experiment Name Argument
    parser.add_argument('--exp_name', default='default_experiment', 
                        help='Name of the experiment (used for saving data files)')
    
    parser.add_argument('--config', default=default_yaml, 
                        help='Name of the yaml file inside the config folder (e.g. "collect_args.yaml")')
    
    args_cli = parser.parse_args()
    
    if '_GAME_RACER_ROOT' not in globals():
        _THIS_DIR = os.path.dirname(os.path.realpath(__file__))
        _GAME_RACER_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))

    config_dir = os.path.join(_GAME_RACER_ROOT, 'config')
    full_config_path = os.path.join(config_dir, args_cli.config)
    
    if not os.path.exists(full_config_path):
        print(f"[ERROR] Config file not found: {full_config_path}")
        sys.exit(1)

    main(full_config_path, args_cli.exp_name)