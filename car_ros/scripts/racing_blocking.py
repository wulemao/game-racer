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

# Safe MPC Import
try:
    from mpc_controller import mpc as mpc_solver_func
except ImportError:
    mpc_solver_func = None
    print("Warning: mpc_controller not found. MPC will not work.")

# --- Global Directory Definitions ---
parent_dir = os.path.dirname(_THIS_DIR)
data_dir = os.path.join(parent_dir, 'data')
os.makedirs(data_dir, exist_ok=True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)

class CarFleet:
    def __init__(self, config):
        self.config = config
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
                
    # --- MATH & PHYSICS HELPERS ---
    def get_curvature(self, x1, y1, x2, y2, x3, y3):
        a = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        b = np.sqrt((x3 - x2)**2 + (y3 - y2)**2)
        c = np.sqrt((x3 - x1)**2 + (y3 - y1)**2)
        s = (a + b + c) / 2
        denom = (a * b * c)
        if denom < 1e-6: return 0.0
        prod = (x2 - x1) * (y3 - y2) - (x3 - x2) * (y2 - y1)
        area_sq = s * (s - a) * (s - b) * (s - c)
        if area_sq < 0: area_sq = 0
        return 4 * np.sqrt(area_sq) / denom * np.sign(prod)

    def calc_shift_factor(self, s, s_opp, vs, vs_opp, sf1=0.4, sf2=0.1, t=1.0):
        """
        Calculates the raw repulsion factor based on TTC.
        """
        if vs == vs_opp: return 0.
        ttc = (s_opp - s) + (vs_opp - vs) * t
        eff_s = ttc 
        factor = sf1 * np.exp(-sf2 * np.abs(eff_s)**2)
        return factor

    def calculate_avoidance_shift(self, s, e, v, opponents, sf1, sf2, blocking_factor, gap, raceline, closest_idx, time_horizon=0.5, track_width=0.44):
        """
        Calculates avoidance shift maintaining original logic logic but scalable to N opponents.
        """
        total_shift = 0.0
        
        # Variables to track the "dominant" opponent (the one creating the strongest urge to shift)
        # This replaces the original logic of "if abs(shift2) > abs(shift1)"
        max_abs_raw_shift = -1.0
        dominant_opp_e = 0.0
        dominant_opp_s = -999.0
        dominant_opp_v = 0.0
        dominant_raw_val = 0.0 # Unsigned raw factor

        raceline_dev = self.cars[0]['waypoint_generator'].raceline_dev
        N_points = len(raceline)

        # --- Phase 1: Accumulate Shifts & Identify Dominant Threat ---
        for (s_opp, e_opp_raw, v_opp) in opponents:
            # 1. Track wrap-around handling
            diff_s = s_opp - s
            if diff_s < -75.: diff_s += 150.
            if diff_s > 75.: diff_s -= 150.
            
            # 2. Get opponent's e_total (absolute lateral pos)
            # Use diff_s for index calculation to match original logic logic
            idx_offset = int(diff_s / gap)
            opp_raceline_idx = (closest_idx + idx_offset) % N_points
            _e_opp_dev = raceline_dev[opp_raceline_idx]
            e_opp_total = e_opp_raw + _e_opp_dev

            # 3. Calculate Raw Shift (Magnitude)
            raw_factor = self.calc_shift_factor(s, s_opp, v, v_opp, sf1, sf2, t=time_horizon)
            
            # 4. Determine Sign (Direction) - Strictly original logic
            # "If I am on left of opponent (e > e_opp), shift positive (left)"
            if e > e_opp_total:
                signed_shift = np.abs(raw_factor)
            else:
                signed_shift = -np.abs(raw_factor)
            
            # 5. Accumulate (Original logic: shift = shift1 + shift2)
            total_shift += signed_shift

            # 6. Track Dominant Opponent
            # equivalent to determining if this opponent is "shift2" or "shift1" in original code
            if np.abs(signed_shift) > max_abs_raw_shift:
                max_abs_raw_shift = np.abs(signed_shift)
                dominant_opp_e = e_opp_total
                dominant_opp_s = s_opp
                dominant_opp_v = v_opp
                dominant_raw_val = raw_factor # Save pure factor for blocking calc

        # If no opponents or negligible shift, return early
        if max_abs_raw_shift <= 1e-6:
            return 0.0

        # --- Phase 2: Alignment Logic (The "shift += e_opp" part) ---
        # Original logic: if (shift + e_opp) * shift < 0: shift = 0
        # This aligns the target to the dominant opponent's lane
        if (total_shift + dominant_opp_e) * total_shift < 0.:
             total_shift = 0.
        else:
             if max_abs_raw_shift > 0.03:
                 total_shift += dominant_opp_e
        
        # --- Phase 3: Blocking Logic (The "bf" part) ---
        # Apply blocking ONLY relative to the dominant opponent (primary threat)
        # Original: if dist_from_opp > 0 ...
        
        dist_from_dom = s - dominant_opp_s
        if dist_from_dom < -75.: dist_from_dom += 150.
        if dist_from_dom > 75.:  dist_from_dom -= 150.

        if dist_from_dom > 0: # If dominant opponent is BEHIND me
            bf = 1.0 - np.exp(-blocking_factor * max(dominant_opp_v - v, 0.))
            
            # Original formula: shift = shift + (e_opp - shift) * bf * factor / sf1
            # Note: We use dominant_raw_val (the result of calc_shift) here
            total_shift = total_shift + (dominant_opp_e - total_shift) * bf * (dominant_raw_val / sf1)

        # --- Phase 4: Boundary Clamping (Your requested optimization) ---
        target_pos = e + total_shift
        
        if target_pos > track_width:
            total_shift = track_width - e
        elif target_pos < -track_width:
            total_shift = -track_width - e
            
        return total_shift

    def get_closest_raceline_idx(self, x, y, raceline, last_i=-1):
        # ... (Keep your optimized implementation) ...
        if last_i == -1:
            # Global search for the closest point on the raceline
            dists = np.sqrt((raceline[:, 0] - x)**2 + (raceline[:, 1] - y)**2)
            return np.argmin(dists)
        else:
            # Local search near the previous index to improve performance
            search_len = 20
            indices = np.arange(last_i, last_i + search_len) % len(raceline)
            raceline_window = raceline[indices]
            dists = np.sqrt((raceline_window[:, 0] - x)**2 + (raceline_window[:, 1] - y)**2)
            local_min = np.argmin(dists)
            return indices[local_min]

    def mpc(self, xyt, pose, *opponents, **kwargs):
        # ... (Same structure as your optimized MPC, calling the new calculate_avoidance_shift) ...
        if mpc_solver_func is None:
            return 0., 0., 0., 0., kwargs.get('last_i', -1)

        s, e, v = pose
        x, y, theta = xyt

        sf1 = kwargs.get('sf1')
        sf2 = kwargs.get('sf2')
        lookahead_factor = kwargs.get('lookahead_factor')
        v_factor = kwargs.get('v_factor')
        blocking_factor = kwargs.get('blocking_factor')
        gap = kwargs.get('gap', 0.06)
        last_i = kwargs.get('last_i', -1)
        wg = self.cars[0]['waypoint_generator']
        
        raceline = wg.raceline
        raceline_dev = wg.raceline_dev
        closest_idx = self.get_closest_raceline_idx(x, y, raceline, last_i)
        _e = raceline_dev[closest_idx]
        e_combined = e + _e
        curr_idx = (closest_idx + 1) % len(raceline)
        next_idx = (curr_idx + 1) % len(raceline)
        traj = []
        dist_target = 0
        # Reformatted list of opponents for the function call
        # Assuming *opponents passes tuples of (s, e, v)
        opp_list = list(opponents)

        for t in np.arange(0.1, 1.05, 0.1): # Predict 1 second horizon, 0.1s steps
            # Accumulate target distance along the track based on speed factor
            dist_target += v_factor * raceline[curr_idx, 2] * 0.1
            shift = self.calculate_avoidance_shift(
                s, e_combined, v, opp_list, sf1, sf2, blocking_factor, gap, raceline, closest_idx, time_horizon=t, track_width=0.44
            )
            # ... (Rest of trajectory generation is identical to your optimized code) ...
            next_dist = np.sqrt((raceline[next_idx, 0] - raceline[curr_idx, 0])**2 +

                                (raceline[next_idx, 1] - raceline[curr_idx, 1])**2)

            while dist_target - next_dist > 0.:
                dist_target -= next_dist
                curr_idx = next_idx
                next_idx = (next_idx + 1) % len(raceline)
                next_dist = np.sqrt((raceline[next_idx, 0] - raceline[curr_idx, 0])**2 +

                                    (raceline[next_idx, 1] - raceline[curr_idx, 1])**2)
            ratio = dist_target / next_dist
            pt = (1. - ratio) * raceline[next_idx, :2] + ratio * raceline[curr_idx, :2]
            theta_traj = np.arctan2(raceline[next_idx, 1] - raceline[curr_idx, 1],
                                    raceline[next_idx, 0] - raceline[curr_idx, 0]) + np.pi / 2.
            shifted_pt = pt + shift * np.array([np.cos(theta_traj), np.sin(theta_traj)])
            traj.append(shifted_pt)
        try:
            throttle, steer = mpc_solver_func([x, y, theta, v], np.array(traj), lookahead_factor=lookahead_factor)
        except Exception:
            throttle, steer = 0.0, 0.0

        # ... (Heuristic correction and Curvature calc - Keep as is) ...
        # (Copying the end of your provided code strictly)
        lookahead_distance = lookahead_factor * raceline[curr_idx, 2]
        lookahead_idx = int(closest_idx + 5) % len(raceline)
        next_idx_h = (lookahead_idx + 1) % len(raceline)
        theta_traj_end = np.arctan2(raceline[next_idx_h, 1] - raceline[lookahead_idx, 1],
                                    raceline[next_idx_h, 0] - raceline[lookahead_idx, 0]) + np.pi / 2.

        alpha = theta - (theta_traj_end - np.pi / 2.)
        if alpha > np.pi: alpha -= 2 * np.pi
        if alpha < -np.pi: alpha += 2 * np.pi

        if np.abs(alpha) > np.pi / 6:
            steer = -np.sign(alpha)
        curv = self.get_curvature(raceline[closest_idx-1, 0], raceline[closest_idx-1, 1],
                                  raceline[closest_idx, 0], raceline[closest_idx, 1],
                                  raceline[(closest_idx+1)%len(raceline), 0], raceline[(closest_idx+1)%len(raceline), 1])

        curv_lookahead = self.get_curvature(raceline[lookahead_idx-1, 0], raceline[lookahead_idx-1, 1],
                                            raceline[lookahead_idx, 0], raceline[lookahead_idx, 1],
                                            raceline[next_idx_h, 0], raceline[next_idx_h, 1])
                                            
        return steer, throttle, curv, curv_lookahead, closest_idx

    # --- CONTROL ALGORITHMS ---
    def pure_pursuit(self, xyt, pose, *opponents, **kwargs):
        s, e, v = pose
        x, y, theta = xyt
        
        sf1 = kwargs.get('sf1')
        sf2 = kwargs.get('sf2')
        lookahead_factor = kwargs.get('lookahead_factor')
        v_factor = kwargs.get('v_factor')
        blocking_factor = kwargs.get('blocking_factor')
        gap = kwargs.get('gap', 0.06)
        last_i = kwargs.get('last_i', -1)

        wg = self.cars[0]['waypoint_generator']
        raceline = wg.raceline
        raceline_dev = wg.raceline_dev
        
        closest_idx = self.get_closest_raceline_idx(x, y, raceline, last_i)
        
        _e = raceline_dev[closest_idx]
        e_combined = e + _e
        
        shift = self.calculate_avoidance_shift(
            s, e_combined, v, opponents, sf1, sf2, blocking_factor, gap, raceline, closest_idx, time_horizon=0.5
        )

        N = len(raceline)
        lookahead_distance = lookahead_factor * raceline[closest_idx, 2]
        lookahead_idx = int(closest_idx + 1 + lookahead_distance // gap) % N
        
        e_lookahead = -raceline_dev[lookahead_idx]
        TRACK_WIDTH_HALF = 0.44
        if e_lookahead + shift > TRACK_WIDTH_HALF:
            shift = TRACK_WIDTH_HALF - e_lookahead
        if e_lookahead + shift < -TRACK_WIDTH_HALF:
            shift = -TRACK_WIDTH_HALF - e_lookahead

        lookahead_point = raceline[lookahead_idx]
        
        # Tangent angle
        next_idx = (lookahead_idx + 1) % N
        prev_idx = (lookahead_idx - 1) % N
        dx_traj = raceline[next_idx, 0] - raceline[lookahead_idx, 0]
        dy_traj = raceline[next_idx, 1] - raceline[lookahead_idx, 1]
        theta_traj = np.arctan2(dy_traj, dx_traj) + np.pi / 2.
        
        shifted_point = lookahead_point + shift * np.array([np.cos(theta_traj), np.sin(theta_traj), 0.])

        # [Fix 3] Calculate Curvature for return
        curv = self.get_curvature(
            raceline[closest_idx-1, 0], raceline[closest_idx-1, 1],
            raceline[closest_idx, 0], raceline[closest_idx, 1],
            raceline[(closest_idx+1)%N, 0], raceline[(closest_idx+1)%N, 1]
        )
        curv_lookahead = self.get_curvature(
            raceline[prev_idx, 0], raceline[prev_idx, 1],
            raceline[lookahead_idx, 0], raceline[lookahead_idx, 1],
            raceline[next_idx, 0], raceline[next_idx, 1]
        )

        v_target = v_factor * lookahead_point[2]
        throttle = (v_target - v) + 9.81 * 0.1 * 4.65 / 20. 

        _dx = shifted_point[0] - x
        _dy = shifted_point[1] - y
        dx_local = _dx * np.cos(theta) + _dy * np.sin(theta)
        dy_local = _dy * np.cos(theta) - _dx * np.sin(theta)
        
        dist_sq = dx_local**2 + dy_local**2
        steer = 2 * self.L_base * dy_local / (dist_sq + 1e-6)
        
        alpha = np.arctan2(dy_local, dx_local)
        if np.abs(alpha) > np.pi / 2:
            steer = np.sign(dy_local)
            
        return steer, throttle, curv, curv_lookahead, closest_idx

    def get_car_state_for_control(self, car_idx):
        c = self.cars[car_idx]
        return (c['s'], c['e'], c['vx'])

    def calculate_collisions(self):
        states = {}
        for i in range(self.num_cars):
            obs = self.cars[i]['env'].state
            states[i] = {'px': obs.x, 'py': obs.y, 'theta': obs.psi, 's': self.cars[i]['s']}

        for i in range(self.num_cars):
            for j in range(i + 1, self.num_cars):
                cost = self.has_collided(states[i], states[j])
                if cost > 0:
                    diff_s = states[j]['s'] - states[i]['s']
                    if diff_s < -75.0: diff_s += 150.0
                    elif diff_s > 75.0: diff_s -= 150.0

                    if diff_s > 0: # j ahead
                        self.apply_decay(i, cost, rear=True)
                        self.apply_decay(j, cost, rear=False)
                    else:
                        self.apply_decay(i, cost, rear=False)
                        self.apply_decay(j, cost, rear=True)

    def apply_decay(self, idx, cost, rear):
        f = 20.0 if rear else 5.0
        decay = np.exp(-f * cost)
        self.cars[idx]['env'].state.vx *= decay
        self.cars[idx]['env'].state.vy *= decay

    def has_collided(self, s1, s2, L=0.18, B=0.12):
        dx = s1['px'] - s2['px']
        dy = s1['py'] - s2['py']
        d_long = dx*np.cos(s1['theta']) + dy*np.sin(s1['theta'])
        d_lat = dy*np.cos(s1['theta']) - dx*np.sin(s1['theta'])
        cost = (np.abs(d_long) - 2*L < 0) * (np.abs(d_lat) - 2*B < 0) * 1.0 
        return cost if cost else 0.0

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

                if mpc_enabled and mpc_solver_func:
                    kwargs['lookahead_factor'] *= 2
                    kwargs['v_factor'] **= 2
                    steer, throttle, curv, curv_lookahead, c['last_i'] = self.fleet.mpc(
                        state_pose, state_track, *opponents, **kwargs
                    )
                else:
                    steer, throttle, curv, curv_lookahead, c['last_i'] = self.fleet.pure_pursuit(
                        state_pose, state_track, *opponents, **kwargs
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