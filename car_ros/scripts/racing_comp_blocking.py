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
import casadi as ca

from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PolygonStamped, Point32
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Float64, Int8
from tf.transformations import quaternion_from_euler, euler_matrix

# --- Path Setup ---
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
from model_arch import SimpleModel

# Safe external MPC Import
try:
    from mpc_controller import mpc as mpc_solver_func
except ImportError:
    mpc_solver_func = None
    print("Warning: mpc_controller not found. Basic MPC will fail.")

# --- Global Directory Definitions ---
CAR_ROS_DIR = os.path.join(_GAME_RACER_ROOT, "car_ros")
P_MODELS_DIR = os.path.join(CAR_ROS_DIR, "p_models")
P_MODELS_REL_DIR = os.path.join(CAR_ROS_DIR, "p_models_rel")
Q_MODELS_DIR = os.path.join(CAR_ROS_DIR, "q_models")
Q_MODELS_REL_DIR = os.path.join(CAR_ROS_DIR, "q_models_rel")

parent_dir = os.path.dirname(_THIS_DIR)
recorded_races_dir = os.path.join(parent_dir, 'recorded_races')
os.makedirs(recorded_races_dir, exist_ok=True)
regrets_dir = os.path.join(parent_dir, 'regrets')
os.makedirs(regrets_dir, exist_ok=True)
n_wins_dir = os.path.join(parent_dir, 'n_wins')
os.makedirs(n_wins_dir, exist_ok=True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)


# ==============================================================================
# 1. IBR Solver (CasADi Implementation)
# ==============================================================================
class IBRSolver:
    def __init__(self, N=10, dt=0.1):
        self.N = N
        self.dt = dt
        self.L = 0.36
        self.mu = 1.6   
        self.g = 9.81   
        
        self.x = ca.MX.sym('x', 4, N+1)  
        self.u = ca.MX.sym('u', 2, N)    
        self.x0_ = ca.MX.sym('x0', 4 + 2*N) 
        self.opp_state_ = ca.MX.sym('target', (N+1)*2) 

        Q = np.diag([0, 0.0, 0.3, 0])
        R = np.diag([1, 1])
        
        cost = 0
        g = []
        
        g.append(self.x[:,0] - self.x0_[:4]) 

        for k in range(N):
            s, e, psi, v = self.x[0,k], self.x[1,k], self.x[2,k], self.x[3,k]
            a, delta = self.u[0,k], self.u[1,k]
            ck = self.x0_[4+k] 

            x_next = ca.vertcat(
                s + (v * ca.cos(psi))/(1-e*ck) * dt,
                e + v * ca.sin(psi) * dt,
                psi + ((v / self.L) * ca.tan(0.36*delta)-v*ck) * dt,
                v + (20./4.65 * a - 1.0) * dt
            )
            g.append(self.x[:, k+1] - x_next)
            
            target_state = np.array([0, 0, 0, 0]) 
            state_error = self.x[:, k] - target_state
            cost += ca.mtimes([state_error.T, Q, state_error])
            cost += ca.mtimes([self.u[:, k].T, R, self.u[:, k]])
            cost += 10 * self.x[1,k]**4 
        
        cost += -10 * (self.x[0, N] - self.x[0,0]) 

        for k in range(N):
            g.append((self.x[0,k] - self.opp_state_[-2*(N+1)+k+1])**2 + 
                     (self.x[1,k] - self.opp_state_[-N-1+k+1])**2)

        for k in range(N):
            g.append((self.x[3,k]**2) * ca.tan(self.u[1,k]))

        for k in range(N):
            g.append(self.x[3,k+1])

        opt_variables = ca.vertcat(ca.reshape(self.x, -1, 1), ca.reshape(self.u, -1, 1))
        opt_constraints = ca.vertcat(*g)
        
        nlp_prob = {'f': cost, 'x': opt_variables, 'g': opt_constraints, 'p': ca.vertcat(self.x0_, self.opp_state_)}
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.max_iter': 100}
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)
        
        total_constraints = 4 + 4*N + N + N + N
        self.lbg = np.zeros((total_constraints, 1)) 
        self.ubg = np.zeros((total_constraints, 1))

    def solve(self, x0_ego, x0_opp, raceline_curvature, N_ibr=3):
        xT_opp = np.array([x0_opp[:2]] * (self.N+1)) 
        xT = np.array([x0_ego[:2]] * (self.N+1))
        u_res = np.zeros(2)
        
        start_lat_accel = 4 + 4*self.N + self.N 
        start_velocity = start_lat_accel + self.N 
        
        for i in range(N_ibr):
            if i % 2 == 0:
                current_x0 = x0_ego
                target_traj = xT_opp
                target_v = x0_opp[3]
            else:
                current_x0 = x0_opp
                target_traj = xT
                target_v = x0_ego[3]
            
            idx_inter = 4 + 4*self.N
            self.lbg[idx_inter : idx_inter+self.N] = 0.04 
            self.ubg[idx_inter : idx_inter+self.N] = 1e9 

            limit_lat = self.mu * self.g * self.L
            self.lbg[start_lat_accel : start_lat_accel+self.N] = -limit_lat
            self.ubg[start_lat_accel : start_lat_accel+self.N] = limit_lat

            self.lbg[start_velocity : start_velocity+self.N] = 0.0
            self.ubg[start_velocity : start_velocity+self.N] = 1.5 * max(target_v, 5.0) 

            curvs = raceline_curvature[:self.N] if len(raceline_curvature) >= self.N else [0.0]*self.N
            params_p = np.concatenate((current_x0, curvs, target_traj.flatten()))
            
            try:
                x_init = np.zeros(4*(self.N+1) + 2*self.N)
                sol = self.solver(x0=x_init, lbg=self.lbg, ubg=self.ubg, p=params_p)
                res = sol['x']
                
                optimal_x = ca.reshape(res[:4*(self.N+1)], 4, self.N+1).full()
                optimal_u = ca.reshape(res[4*(self.N+1):], 2, self.N).full()
                
                if i % 2 == 0:
                    xT = optimal_x[:2, :].T
                    u_res = optimal_u[:, 0]
                else:
                    xT_opp = optimal_x[:2, :].T
            except Exception as e:
                pass
        
        heading_error = x0_opp[2] 
        if heading_error > 0.5:
            u_res[1] = -1.0 
            u_res[0] = 0.5 
        if heading_error < -0.5:
            u_res[1] = 1.0 
            u_res[0] = 0.5 
                
        return u_res


# ==============================================================================
# 2. Parameter Optimizer (Fixed for Global Tensor Order)
# ==============================================================================
class ParameterOptimizer:
    def __init__(self, config, model, V_models, opponent_models, input_dim, num_cars):
        self.params = config['optimization_params']
        self.style = config['argparse']['opt_style']
        self.grad_rate = float(self.params['grad_rate'])
        self.steps = int(self.params['steps_per_iter'])
        self.k_linear = int(self.params['k_linear'])
        
        self.model = model
        self.V_models = V_models 
        self.opponent_models = opponent_models
        self.num_cars = num_cars
        self.regrets = [[] for _ in range(num_cars)]
        self.EP_LEN = 500

    def get_optimization_horizon(self, strategy_name, is_ego=False):
        if is_ego: return 450 
        S = 50 
        if 'ours-low_p' in strategy_name: S = 10
        elif 'ours-high_p' in strategy_name: S = 250
        return self.EP_LEN - S

    def optimize(self, fleet, construct_tensor_fn, current_step):
        """
        Main optimization loop.
        Logic restored: 
        1. 'model' optimization runs if style is grad/linear (Active Control).
        2. 'V_model' optimization runs ALWAYS (Logging/Regrets), but only updates Fleet if style is value_opt.
        """
        
        # --- Loop over all cars (Ego + Opponents) ---
        for k in range(self.num_cars):
            # Check if this car is within its optimization time window
            strategy = fleet.cars[k]['strategy']
            is_ego = (k == 0)
            horizon = self.get_optimization_horizon(strategy, is_ego)
            
            if current_step >= horizon:
                continue

            # Identify models for this car
            # For Ego (k=0): uses self.model and self.V_models[0]
            # For Opponent (k>0): uses self.opponent_models[k-1] and self.V_models[k]
            
            main_model = None
            if is_ego:
                main_model = self.model
            elif k-1 < len(self.opponent_models):
                main_model = self.opponent_models[k-1]
                
            v_model = self.V_models[k] if k < len(self.V_models) else None

            # --- PHASE 1: Active Control Optimization (Regret Model) ---
            # Updates parameters if style is 'grad' or 'linear'
            if self.style in ['grad', 'linear'] and main_model:
                # Construct fresh tensor for control optimization
                X_control = construct_tensor_fn(viewer_idx=k)
                
                if self.style == 'grad':
                    X_control = X_control.clone().detach().requires_grad_(True)
                    self._optimize_grad_step(X_control, main_model, fleet, viewer_idx=k, update_fleet=True)
                
                elif self.style == 'linear':
                    self._optimize_linear_step(X_control, main_model, fleet, viewer_idx=k, update_fleet=True)

            # --- PHASE 2: Value Function Optimization (Logging / Alternative Control) ---
            # Always runs to populate 'regrets' list.
            # Only updates Fleet if style is 'value_opt'.
            if v_model:
                # Construct fresh tensor for V optimization (isolating gradients from Phase 1)
                X_value = construct_tensor_fn(viewer_idx=k)
                X_value = X_value.clone().detach().requires_grad_(True)
                
                should_update_fleet = (self.style == 'value_opt')
                
                self._optimize_grad_step(
                    X_value, 
                    v_model, 
                    fleet, 
                    viewer_idx=k, 
                    update_fleet=should_update_fleet,
                    log_regret_list=self.regrets[k]
                )

    def _optimize_grad_step(self, X, model, fleet, viewer_idx, update_fleet=True, log_regret_list=None):
        """
        Performs Gradient Descent.
        Args:
            update_fleet (bool): If True, writes optimized params back to CarFleet.
            log_regret_list (list): If provided, appends regret stats (new_val - prev_val).
        """
        physics_dim = 8 * self.num_cars
        param_start_idx = physics_dim 
        
        # Initial value for regret calculation
        prev_val = 0.0
        if log_regret_list is not None:
            prev_val = float(model(X)[0,0].item())

        # Gradient Descent Loop
        for _ in range(self.steps):
            X_curr = torch.autograd.Variable(X, requires_grad=True)
            model.zero_grad()
            preds = model(X_curr)
            grad = torch.autograd.grad(preds[0, 0], X_curr, retain_graph=True)[0].data
            
            total_param_count = 5 * self.num_cars
            for i in range(total_param_count):
                idx = param_start_idx + i
                val = X_curr[0, idx].item() + self.grad_rate * grad[0, idx].item()
                X[0, idx] = val 

        # Logging Logic (Matches original code structure)
        if log_regret_list is not None:
            new_val = float(model(X)[0,0].item())
            # Log specific params (Lookahead and Speed of the Viewer)
            # Viewer's params are always the first block (index 0 relative to param_start)
            base = param_start_idx
            lookahead_val = float(X[0, base+2].item())
            speed_val = float(X[0, base+3].item())
            log_regret_list.append([new_val - prev_val, new_val, prev_val, lookahead_val, speed_val])

        # Write Back Logic (Conditional)
        if update_fleet:
            self._write_back_params(X, fleet, viewer_idx, param_start_idx)

    def _optimize_linear_step(self, X, model, fleet, viewer_idx, update_fleet=True):
        physics_dim = 8 * self.num_cars
        param_start_idx = physics_dim
        total_param_count = 5 * self.num_cars
        
        indices = list(range(total_param_count))
        np.random.shuffle(indices)
        
        for i in indices:
            idx = param_start_idx + i
            param_type = i % 5 
            
            if param_type in [0, 1]: 
                vals = np.linspace(0.1, 0.5, self.k_linear)
            elif param_type == 2: 
                vals = np.linspace(0.15, 0.5, self.k_linear)
            elif param_type == 3: 
                vals = np.linspace(0.85, 1.1, self.k_linear)
            else: 
                vals = np.linspace(0., 1., self.k_linear)
            
            X_batch = X.repeat(self.k_linear, 1)
            X_batch[:, idx] = torch.tensor(vals).float().to(DEVICE)
            
            with torch.no_grad():
                preds = model(X_batch)[:, 0]
            
            best_idx = torch.argmax(preds)
            best_val = vals[best_idx]
            X[0, idx] = best_val
            
        if update_fleet:
            self._write_back_params(X, fleet, viewer_idx, param_start_idx)

    def _write_back_params(self, X, fleet, viewer_idx, param_start_idx):
        """Writes tensor values back to Fleet."""
        current_tensor_idx = param_start_idx
        
        # Viewer's view of global indices:
        # X's param blocks are ordered: [Viewer_Real, Belief_about_(Viewer+1), Belief_about_(Viewer+2)...]
        # We need to map these blocks back to the correct fleet targets.
        
        # Indices in X correspond to this permutation:
        permuted_indices = [(viewer_idx + k) % fleet.num_cars for k in range(fleet.num_cars)]
        
        for k, global_car_id in enumerate(permuted_indices):
            # k is the block index in Tensor X (0, 1, 2...)
            # global_car_id is the actual car ID in the fleet
            
            if k == 0:
                # Block 0 is ALWAYS Viewer's Real Params
                self._update_params_from_tensor_indices(
                    fleet.cars[viewer_idx]['opt_params'], X, current_tensor_idx
                )
            else:
                # Block > 0 is Viewer's Belief about global_car_id
                self._update_params_from_tensor_indices(
                    fleet.cars[viewer_idx]['belief_params'][global_car_id], X, current_tensor_idx
                )
            
            current_tensor_idx += 5

    def _update_params_from_tensor_indices(self, params_dict, X, start_idx):
        params_dict['sf1'] = max(0.1, min(0.5, X[0, start_idx].item()))
        params_dict['sf2'] = max(0.1, min(0.5, X[0, start_idx+1].item()))
        params_dict['lookahead'] = max(0.15, min(0.5, X[0, start_idx+2].item()))
        params_dict['speed'] = max(0.85, min(1.1, X[0, start_idx+3].item()))
        params_dict['blocking'] = max(0., min(1., X[0, start_idx+4].item()))

# ==============================================================================
# 3. Car Fleet
# ==============================================================================
class CarFleet:
    def __init__(self, config, vis_enabled=True):
        self.config = config
        self.vis_enabled = vis_enabled 
        if 'num_cars' not in config['argparse']:
            raise KeyError("Config Error: 'num_cars' parameter is missing.")

        self.num_cars = int(config['argparse']['num_cars'])
        raw_opp = config['argparse']['opp_method']
        self.opp_strategies = [raw_opp] if isinstance(raw_opp, str) else raw_opp
        self.strategies = ['ours'] + self.opp_strategies
    
        self.ep_no = 0
        self.n_wins = [0] * self.num_cars
        self.dataset = []
        self.buffer = []
        self.cars = {}
        self.publishers = {}
        
        if 'ibr' in self.strategies:
            self.ibr_solver = IBRSolver(N=10, dt=config['car_dynamics']['DT'])
        
        dyn = config['car_dynamics']
        self.DT = dyn['DT']
        self.DELAY = dyn['DELAY']
        self.H = dyn['H']
        self.traj_type = config['sim']['numerical']['trajectory_type']
        self.traj_type = os.path.join(SIM_PARAMS_PATH, self.traj_type)
        self.sim_type = config['sim']['type']
        
        self.LF = 0.12 
        self.LR = 0.24 
        self.L_base = self.LF + self.LR

        self.init_car_params()
        self.init_env()
        self.init_publishers()

    def init_publishers(self):
        if not self.vis_enabled: return
        
        for i in range(self.num_cars):
            name_idx = f"_{i}" if i > 0 else ""
            self.publishers[i] = {
                'path_nn': rospy.Publisher(f'path_nn{name_idx}', Path, queue_size=1),
                'pose': rospy.Publisher(f'pose{name_idx}', PoseWithCovarianceStamped, queue_size=1),
                'odom': rospy.Publisher(f'odom{name_idx}', Odometry, queue_size=1),
                'body': rospy.Publisher(f'body{name_idx}', PolygonStamped, queue_size=1),
                'ref_traj': rospy.Publisher(f'ref_trajectory{name_idx}', Path, queue_size=1),
                'throttle': rospy.Publisher(f'throttle{name_idx}', Float64, queue_size=1),
                'steer': rospy.Publisher(f'steer{name_idx}', Float64, queue_size=1),
            }
            if i == 0:
                self.publishers[i]['waypoint_list'] = rospy.Publisher('waypoint_list', Path, queue_size=1)
                self.publishers[i]['left_boundary'] = rospy.Publisher('left_boundary', Path, queue_size=1)
                self.publishers[i]['right_boundary'] = rospy.Publisher('right_boundary', Path, queue_size=1)
                self.publishers[i]['raceline'] = rospy.Publisher('raceline', Path, queue_size=1)
                self.publishers[i]['status'] = rospy.Publisher('status', Int8, queue_size=1)

    def init_car_params(self):
        default_params = {'sf1': 0.4, 'sf2': 0.2, 'lookahead': 0.35, 'speed': 1.0, 'blocking': 0.}
        for i in range(self.num_cars):
            self.cars[i] = {
                'index': i,
                'strategy': self.strategies[i],
                # Legacy fields
                'curr_speed_factor': 1.0, 'curr_lookahead_factor': 0.35,
                'curr_sf1': 0.4, 'curr_sf2': 0.2, 'blocking': 0.0,
                'last_i': -1,
                'curr_steer': 0.0,
                's': 0.0, 'e': 0.0, 'vx': 0.0, 'px': 0.0, 'py': 0.0, 'psi': 0.0,
                'obs': [],
                'action': np.array([0.0, 0.0]),
                'target_pos_tensor': [],
                'lookaheads': [],
                'curv': 0.0, 'curv_lookahead': 0.0, 'theta_diff': 0.0,
                
                'opt_params': default_params.copy(),
                'belief_params': {} 
            }
            for j in range(self.num_cars):
                if i != j:
                    self.cars[i]['belief_params'][j] = default_params.copy()

    def init_env(self):
        self.start_poses = [
            [3., 5., -np.pi/2.-0.72],
            [0., 0., -np.pi/2.-0.5],
            # [-2., -6., -np.pi/2.-0.5],
            # [3., 8., -np.pi/2.-0.72],
            # [-3., -9., -np.pi/2.-0.5]
        ]
        
        if self.num_cars > len(self.start_poses):
            extra = self.num_cars - len(self.start_poses)
            for i in range(extra):
                base = self.start_poses[i % len(self.start_poses)]
                self.start_poses.append([base[0], base[1]-10.0, base[2]])

        for i in range(self.num_cars):
            self.cars[i]['model_params'] = DynamicParams(num_envs=1, DT=self.DT, Sa=0.34, Sb=-0., Ta=20., Tb=.0, mu=0.5, delay=self.DELAY)
            self.cars[i]['dynamic_single'] = DynamicBicycleModel(self.cars[i]['model_params'])
            self.cars[i]['dynamic_single'].reset()
            
            h_factor = 2. if i == 0 else 1.
            self.cars[i]['waypoint_generator'] = WaypointGenerator(self.traj_type, self.DT, self.H, h_factor)
            
            if self.sim_type == 'numerical':
                self.cars[i]['env'] = OffroadCar({}, self.cars[i]['dynamic_single'])

    def reset_episode(self):
        self.ep_no += 1
        
        if self.ep_no < 34: order = [0, 1, 2]
        elif self.ep_no < 67: order = [1, 2, 0]
        else: order = [2, 0, 1]

        if self.num_cars > 3:
            remaining = [i for i in range(len(self.start_poses)) if i not in order]
            order.extend(remaining)

        ego_pose_idx = order[0]
        available_slots = [idx for idx in range(len(self.start_poses)) if idx != ego_pose_idx]
        opponent_slots = sorted(available_slots, reverse=True) 

        for i in range(self.num_cars):
            if i == 0: pose_idx = ego_pose_idx
            else: pose_idx = opponent_slots[(i - 1) % len(opponent_slots)]
            
            pose = self.start_poses[pose_idx]
            self.cars[i]['obs'] = self.cars[i]['env'].reset(pose=pose)
            self.cars[i]['last_i'] = -1
            self.cars[i]['curr_steer'] = 0.0
            self.cars[i]['waypoint_generator'].last_i = -1
            
            if self.ep_no % self.num_cars == i:
                c = self.cars[i]
                c['curr_sf1'] = np.random.uniform(0.1, 0.5)
                c['curr_sf2'] = np.random.uniform(0.1, 0.5)
                c['curr_lookahead_factor'] = np.random.uniform(0.12, 0.5)
                c['curr_speed_factor'] = np.random.uniform(0.85, 1.1)
                c['blocking'] = np.random.uniform(0., 1.0)
                
                c['opt_params'].update({
                    'sf1': c['curr_sf1'], 'sf2': c['curr_sf2'], 
                    'lookahead': c['curr_lookahead_factor'], 
                    'speed': c['curr_speed_factor'], 'blocking': c['blocking']
                })
                print(f"Episode {self.ep_no}: Randomized Car {i}")

    def get_curvature(self, raceline, idx):
        N = len(raceline)
        p1 = raceline[idx-1]
        p2 = raceline[idx]
        p3 = raceline[(idx+1)%N]
        
        a = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        b = np.sqrt((p3[0] - p2[0])**2 + (p3[1] - p2[1])**2)
        c = np.sqrt((p3[0] - p1[0])**2 + (p3[1] - p1[1])**2)
        s = (a + b + c) / 2
        denom = (a * b * c)
        if denom == 0: return 0.0
        
        prod = (p2[0] - p1[0]) * (p3[1] - p2[1]) - (p3[0] - p2[0]) * (p2[1] - p1[1])
        area_sq = s * (s - a) * (s - b) * (s - c)
        if area_sq < 0: area_sq = 0
        return 4 * np.sqrt(area_sq) / denom * np.sign(prod)

    def calc_shift(self, s, s_opp, vs, vs_opp, sf1=0.4, sf2=0.1, t=1.0):
        if vs == vs_opp: return 0.
        ttc = (s_opp - s) + (vs_opp - vs) * t
        factor = sf1 * np.exp(-sf2 * np.abs(ttc)**2)
        return factor

    # --- Control Logic ---
    def get_control_for_car(self, car_idx):
        c = self.cars[car_idx]
        strategy = c['strategy']
        
        px, py, psi = c['px'], c['py'], c['psi']
        s, e, v = c['s'], c['e'], c['vx']
        xyt = (px, py, psi)
        pose = (s, e, v)

        # Prepare Opponent States
        opponents_states = []
        for i in range(self.num_cars):
            if i != car_idx:
                opp_state = self.get_mpc_state(i) # (s, e, v)
                opponents_states.append(opp_state)

        p = c['opt_params']
        params = {
            'sf1': p['sf1'],
            'sf2': p['sf2'],
            'lookahead_factor': p['lookahead'],
            'v_factor': p['speed'],
            'blocking_factor': p['blocking'],
            'gap': 0.06,
            'last_i': c['last_i'],
            'curv': c['curv'],
            'curv_lookahead': c['curv_lookahead']
        }

        if strategy == 'ibr':
            if len(opponents_states) > 0:
                opp_state = opponents_states[0]
                N_ibr = self.ibr_solver.N
                raw_curvs = c.get('lookaheads', [0.0] * N_ibr)
                curv_list = raw_curvs[:N_ibr] if len(raw_curvs) >= N_ibr else raw_curvs + [0.0]*(N_ibr - len(raw_curvs))

                u = self.ibr_solver.solve(
                    np.array([s, e, c['theta_diff'], v]),
                    np.array([opp_state[0], opp_state[1], 0.0, opp_state[2]]), 
                    curv_list
                )
                return u[1], u[0], 0., 0., c['last_i']
            else:
                strategy = 'mpc' 

        if strategy == 'mpc' or (self.config['argparse']['mpc_enabled'] and mpc_solver_func is not None):
            return self.mpc(xyt, pose, opponents_states, **params)

        return self.pure_pursuit(xyt, pose, opponents_states, **params)

    def get_closest_raceline_idx(self, x, y, raceline, last_i):
        if last_i != -1:
            N = len(raceline)
            indices = [ (last_i + k) % N for k in range(20) ]
            local_pts = raceline[indices]
            dists = np.sqrt((local_pts[:,0]-x)**2 + (local_pts[:,1]-y)**2)
            min_local_idx = np.argmin(dists)
            return indices[min_local_idx]
        else:
            dists = np.sqrt((raceline[:,0]-x)**2 + (raceline[:,1]-y)**2)
            return np.argmin(dists)

    def calculate_avoidance_shift(self, s, e_combined, v, opponents_states, sf1, sf2, 
                                  blocking_factor, gap, raceline, closest_idx, time_horizon):
        shift = 0.0
        raw_shifts = []
        
        for (opp_s, opp_e, opp_v) in opponents_states:
            val = self.calc_shift(s, opp_s, v, opp_v, sf1, sf2, time_horizon)
            
            if e_combined > opp_e:
                val = np.abs(val)
            else:
                val = -np.abs(val)
            raw_shifts.append(val)

        shift = sum(raw_shifts)
        
        if len(raw_shifts) == 2:
            if abs(raw_shifts[1]) > abs(raw_shifts[0]):
                opp_e = opponents_states[0][1]
                if (shift + opp_e) * shift < 0:
                     shift = 0.

        closest_dist = -float('inf')
        target_opp_idx = -1
        for idx, (opp_s, _, opp_v) in enumerate(opponents_states):
            dist = s - opp_s
            if dist < -75.: dist += 150.
            if dist > 75.: dist -= 150.
            if dist > 0 and dist > closest_dist:
                closest_dist = dist
                target_opp_idx = idx

        if target_opp_idx != -1:
            (opp_s, opp_e, opp_v) = opponents_states[target_opp_idx]
            bf = 1 - np.exp(-blocking_factor * max(opp_v - v, 0.))
            shift_block = self.calc_shift(s, opp_s, v, opp_v, sf1, sf2, time_horizon)
            shift = shift + (opp_e - shift) * bf * shift_block / sf1

        return shift

    def pure_pursuit(self, xyt, pose, opponents_states, **kwargs):
        s, e, v = pose
        x, y, theta = xyt
        
        sf1, sf2 = kwargs.get('sf1'), kwargs.get('sf2')
        lookahead_factor = kwargs.get('lookahead_factor')
        v_factor = kwargs.get('v_factor')
        blocking_factor = kwargs.get('blocking_factor')
        gap = kwargs.get('gap', 0.06)
        last_i = kwargs.get('last_i', -1)

        wg = self.cars[0]['waypoint_generator']
        raceline = wg.raceline
        raceline_dev = wg.raceline_dev
        N = len(raceline)
        
        closest_idx = self.get_closest_raceline_idx(x, y, raceline, last_i)
        _e = raceline_dev[closest_idx]
        e_combined = e + _e
        
        shift = self.calculate_avoidance_shift(
            s, e_combined, v, opponents_states, sf1, sf2, 
            blocking_factor, gap, raceline, closest_idx, time_horizon=0.5
        )

        lookahead_distance = lookahead_factor * raceline[closest_idx, 2]
        lookahead_idx = int(closest_idx + 1 + lookahead_distance // gap) % N
        
        e_lookahead = -raceline_dev[lookahead_idx]
        TRACK_WIDTH_HALF = 0.44
        if e_lookahead + shift > TRACK_WIDTH_HALF: shift = TRACK_WIDTH_HALF - e_lookahead
        if e_lookahead + shift < -TRACK_WIDTH_HALF: shift = -TRACK_WIDTH_HALF - e_lookahead

        lookahead_point = raceline[lookahead_idx]
        
        next_idx = (lookahead_idx + 1) % N
        dx_traj = raceline[next_idx, 0] - raceline[lookahead_idx, 0]
        dy_traj = raceline[next_idx, 1] - raceline[lookahead_idx, 1]
        theta_traj = np.arctan2(dy_traj, dx_traj) + np.pi / 2.
        
        shifted_point = lookahead_point[:2] + shift * np.array([np.cos(theta_traj), np.sin(theta_traj)])

        _dx = shifted_point[0] - x
        _dy = shifted_point[1] - y
        dx_local = _dx * np.cos(theta) + _dy * np.sin(theta)
        dy_local = _dy * np.cos(theta) - _dx * np.sin(theta)
        
        dist_sq = dx_local**2 + dy_local**2
        steer = 2 * self.L_base * dy_local / (dist_sq + 1e-6)
        
        v_target = v_factor * lookahead_point[2]
        throttle = (v_target - v) + 0.5 

        curv = self.get_curvature(raceline, closest_idx)
        curv_la = self.get_curvature(raceline, lookahead_idx)

        return steer, throttle, curv, curv_la, closest_idx

    def mpc(self, xyt, pose, opponents_states, **kwargs):
        if mpc_solver_func is None:
            return 0., 0., 0., 0., kwargs.get('last_i', -1)

        s, e, v = pose
        x, y, theta = xyt
        
        sf1, sf2 = kwargs.get('sf1'), kwargs.get('sf2')
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
        
        traj = []
        dist_target = 0
        curr_idx = (closest_idx + 1) % len(raceline)
        next_idx = (curr_idx + 1) % len(raceline)

        for t in np.arange(0.1, 1.1, 0.1): 
            dist_target += v_factor * raceline[curr_idx, 2] * 0.1
            
            shift = self.calculate_avoidance_shift(
                s, e_combined, v, opponents_states, sf1, sf2, 
                blocking_factor, gap, raceline, closest_idx, time_horizon=t
            )
            
            next_dist = np.sqrt((raceline[next_idx, 0] - raceline[curr_idx, 0])**2 + 
                                (raceline[next_idx, 1] - raceline[curr_idx, 1])**2)
            while dist_target - next_dist > 0.:
                dist_target -= next_dist
                curr_idx = next_idx
                next_idx = (next_idx + 1) % len(raceline)
                next_dist = np.sqrt((raceline[next_idx, 0] - raceline[curr_idx, 0])**2 + 
                                    (raceline[next_idx, 1] - raceline[curr_idx, 1])**2)
            
            ratio = dist_target / next_dist
            pt = (1. - ratio) * raceline[curr_idx, :2] + ratio * raceline[next_idx, :2]
            
            theta_traj = np.arctan2(raceline[next_idx, 1] - raceline[curr_idx, 1], 
                                    raceline[next_idx, 0] - raceline[curr_idx, 0]) + np.pi / 2.
            shifted_pt = pt + shift * np.array([np.cos(theta_traj), np.sin(theta_traj)])
            traj.append(shifted_pt)

        try:
            throttle, steer = mpc_solver_func([x, y, theta, v], np.array(traj), lookahead_factor=lookahead_factor)
        except:
            throttle, steer = 0.0, 0.0

        curv = self.get_curvature(raceline, closest_idx)
        curv_lookahead = self.get_curvature(raceline, curr_idx) 
        
        return steer, throttle, curv, curv_lookahead, closest_idx

    def calculate_collisions(self):
        states = {}
        for i in range(self.num_cars):
            obs = self.cars[i]['env'].state
            states[i] = {'px': obs.px, 'py': obs.py, 'theta': obs.psi, 's': self.cars[i]['s']}

        for i in range(self.num_cars):
            for j in range(i + 1, self.num_cars):
                cost = self.has_collided(states[i], states[j])
                if cost > 0:
                    diff_s = states[j]['s'] - states[i]['s']
                    if diff_s < -75.0: diff_s += 150.0
                    elif diff_s > 75.0: diff_s -= 150.0
                    
                    if diff_s > 0: 
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
    
    def get_mpc_state(self, idx):
        c = self.cars[idx]
        return (c['s'], c['e'], c['vx'])

    def save_data(self, exp_name, regrets_dict):
        print(f"Saving Dataset to {exp_name}...")
        
        with open(os.path.join(regrets_dir, f'{exp_name}.pkl'), 'wb') as f:
            pickle.dump(regrets_dict, f)
        with open(os.path.join(recorded_races_dir, f'{exp_name}.pkl'), 'wb') as f:
            pickle.dump(np.array(self.dataset), f)
        with open(os.path.join(n_wins_dir, f'{exp_name}.txt'), 'w') as f:
            for item in self.n_wins:
                f.write("%s\n" % item)

    def _pub_path_helper(self, pub, points, timestamp):
        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = timestamp
        for i in range(len(points)):
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = float(points[i][0])
            pose.pose.position.y = float(points[i][1])
            path.poses.append(pose)
        pub.publish(path)

    def publish_static_map(self):
        if not self.vis_enabled: return
        
        wg = self.cars[0]['waypoint_generator']
        now = rospy.Time.now()
        
        self._pub_path_helper(self.publishers[0]['raceline'], wg.raceline, now)
        self._pub_path_helper(self.publishers[0]['left_boundary'], wg.left_boundary, now)
        self._pub_path_helper(self.publishers[0]['right_boundary'], wg.right_boundary, now)

    def visualize_all(self, timestamp):
        if not self.vis_enabled: return

        for k in range(self.num_cars):
            c = self.cars[k]
            pubs = self.publishers[k]
            
            q = quaternion_from_euler(0, 0, c['psi'])
            pose = PoseWithCovarianceStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = timestamp
            pose.pose.pose.position.x = c['px']
            pose.pose.pose.position.y = c['py']
            pose.pose.pose.orientation.x = q[0]
            pose.pose.pose.orientation.y = q[1]
            pose.pose.pose.orientation.z = q[2]
            pose.pose.pose.orientation.w = q[3]
            pubs['pose'].publish(pose)
            
            odom = Odometry()
            odom.header.frame_id = 'map'
            odom.header.stamp = timestamp
            odom.pose.pose = pose.pose.pose
            odom.twist.twist.linear.x = c['vx']
            odom.twist.twist.linear.y = c['vy']
            odom.twist.twist.angular.z = c['omega']
            pubs['odom'].publish(odom)
            
            pts = np.array([
                [self.LF, self.L_base/3], [self.LF, -self.L_base/3],
                [-self.LR, -self.L_base/3], [-self.LR, self.L_base/3]
            ])
            R = euler_matrix(0, 0, c['psi'])[:2, :2]
            pts = np.dot(R, pts.T).T + np.array([c['px'], c['py']])
            
            body = PolygonStamped()
            body.header.frame_id = 'map'
            body.header.stamp = timestamp
            for pt in pts:
                p = Point32()
                p.x, p.y = float(pt[0]), float(pt[1])
                body.polygon.points.append(p)
            pubs['body'].publish(body)
            
            pubs['throttle'].publish(Float64(c['action'][0]))
            pubs['steer'].publish(Float64(c['action'][1]))
            
            if len(c['target_pos_tensor']) > 0:
                traj_pts = np.array(c['target_pos_tensor'])
                self._pub_path_helper(pubs['ref_traj'], traj_pts, timestamp)


# ==============================================================================
# 4. CarNode
# ==============================================================================
class CarNode:
    def __init__(self, config_path):
        rospy.init_node('car_node_comp')
        self.config = self.load_config(config_path)
        
        args = self.config['argparse']
        self.exp_name = args['exp_name']
        
        opt_style = args['opt_style']
        self.exp_name = f"race_exp_{opt_style}" 
        if args['mpc_enabled']: self.exp_name += '_mpc'
        if args['use_rel']: self.exp_name += '_rel'
        
        vis_enabled = self.config.get('sim', {}).get('vis', True)
        self.fleet = CarFleet(self.config, vis_enabled)
        
        self.init_models_selectively() 
        
        self.i = 0
        self.ep_len = self.config['car_dynamics']['EP_LEN']
        
        dt = self.config['car_dynamics']['DT']
        self.timer_ = rospy.Timer(rospy.Duration(dt/3.0), self.timer_callback)
        self.slow_timer_ = rospy.Timer(rospy.Duration(10.0), self.slow_timer_callback)

    @staticmethod
    def load_config(path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def init_models_selectively(self):
        strategies = self.fleet.strategies 
        
        state_dim = 13
        input_dim = state_dim * self.fleet.num_cars
        fs = 128
        
        args = self.config['argparse']
        suffix = "_mpc" if args['mpc_enabled'] else ""
        p_dir = P_MODELS_REL_DIR if args['use_rel'] else P_MODELS_DIR
        q_dir = Q_MODELS_REL_DIR if args['use_rel'] else Q_MODELS_DIR

        self.model = None
        if any('ours' in s for s in strategies):
            self.model = SimpleModel(input_dim, [self.fleet.num_cars*fs, self.fleet.num_cars*fs, self.fleet.num_cars*64], 1)
            try:
                self.model.load_state_dict(torch.load(os.path.join(p_dir, f"model_multi_myopic{suffix}.pth")))
                self.model.eval()
            except Exception as e:
                print(f"Main Model Load Warning: {e}")
            
        self.V_models = []
        for k in range(self.fleet.num_cars):
            if 'ours' in strategies[k]:
                v_net = SimpleModel(input_dim, [128, 128, 64], 1)
                try:
                    path = os.path.join(q_dir, f"model_multi{k}_myopic{suffix}.pth")
                    v_net.load_state_dict(torch.load(path))
                    v_net.eval()
                    self.V_models.append(v_net)
                except Exception as e:
                    self.V_models.append(None)
            else:
                self.V_models.append(None)

        self.opponent_models = []
        for i, strat in enumerate(self.fleet.opp_strategies): 
            model = None
            if 'ours' in strat: 
                model = SimpleModel(input_dim, [self.fleet.num_cars*fs, self.fleet.num_cars*fs, self.fleet.num_cars*64], 1)
                path = None
                if strat == "ours-low_data":
                    path = os.path.join(p_dir, f"model_multi_small_myopic{suffix}.pth")
                elif strat == "ours-low_p":
                    path = os.path.join(p_dir, f"model_multi_myopic_s{suffix}.pth")
                
                if path:
                    try:
                        model.load_state_dict(torch.load(path))
                        model.eval()
                    except:
                        pass
            self.opponent_models.append(model)
            
        self.optimizer = ParameterOptimizer(
            self.config, self.model, self.V_models, self.opponent_models, input_dim, self.fleet.num_cars
        )

    def timer_callback(self, event):
        try:
            self.i += 1
            if self.i > self.ep_len:
                self.handle_episode_end()
                return

            now = rospy.Time.now()

            # 1. Update State
            for k in range(self.fleet.num_cars):
                c = self.fleet.cars[k]
                obs_tensor = jnp.array(c['obs'][:5])
                target_pos, _, s, e = c['waypoint_generator'].generate(
                    obs_tensor, dt=self.config['car_dynamics']['DT_torch'], mu_factor=1.0, body_speed=c['obs'][3]
                )
                c['s'], c['e'] = s, e
                c['target_pos_tensor'] = target_pos
                c['lookaheads'] = target_pos[:, 3].tolist()
                c['curv'] = float(target_pos[0, 3])
                c['curv_lookahead'] = float(target_pos[-1, 3])
                c['theta'] = target_pos[0, 2]
                
                full = self.fleet.get_obs_state(k).tolist()
                c['px'], c['py'], c['psi'], c['vx'], c['vy'], c['omega'] = full
                c['theta_diff'] = np.arctan2(np.sin(c['theta']-c['psi']), np.cos(c['theta']-c['psi']))

            # 2. Optimization (Executes logic for Ego and All Opponents based on timing)
            self.optimizer.optimize(self.fleet, self.construct_state_tensor, self.i)
            
            # Sync back to control variables
            for k in range(self.fleet.num_cars):
                if 'ours' in self.fleet.cars[k]['strategy']:
                    c = self.fleet.cars[k]
                    p = c['opt_params']
                    c['curr_sf1'] = p['sf1']
                    c['curr_sf2'] = p['sf2']
                    c['curr_lookahead_factor'] = p['lookahead']
                    c['curr_speed_factor'] = p['speed']
                    c['blocking'] = p['blocking']

            # 3. Control Loop
            for k in range(self.fleet.num_cars):
                steer, throttle, curv, curv_la, last_i = self.fleet.get_control_for_car(k)
                
                if self.i < 6:
                    throttle = 0.0
                    steer = 0.0
                
                c = self.fleet.cars[k]
                c['last_i'] = last_i

                # Stabilization Logic
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
            
            # 4. Physics & Data Logging
            self.fleet.calculate_collisions()
            self.collect_data()

            # 5. Visualization
            self.fleet.visualize_all(now)

        except Exception as e:
            traceback.print_exc()

    def construct_state_tensor(self, viewer_idx=0):
        """
        Constructs state tensor X for a specific 'viewer' (Car k).
        Order: [Physics] + [Viewer_Real_Params] + [Viewer_Belief_Opp1] + [Viewer_Belief_Opp2]...
        """
        vec = []
        
        # --- Part A: Physics State (Shared Ground Truth) ---
        for k in range(self.fleet.num_cars): vec.append(self.fleet.cars[k]['s'])
        for k in range(self.fleet.num_cars): vec.append(self.fleet.cars[k]['e'])
        for k in range(self.fleet.num_cars):
            c = self.fleet.cars[k]
            vec.extend([c['theta_diff'], c['vx'], c['vy'], c['omega']])
        for k in range(self.fleet.num_cars): vec.append(self.fleet.cars[k]['curv'])
        for k in range(self.fleet.num_cars): vec.append(self.fleet.cars[k]['curv_lookahead'])
        
        # --- Part B: Parameters (Perspective Dependent) ---
        # 1. Viewer's Own Real Parameters (First Block)
        p_self = self.fleet.cars[viewer_idx]['opt_params']
        vec.extend([p_self['sf1'], p_self['sf2'], p_self['lookahead'], p_self['speed'], p_self['blocking']])
        
        # 2. Viewer's Beliefs about others (Subsequent Blocks)
        # Order is strictly by car index, skipping self
        for k in range(self.fleet.num_cars):
            if k != viewer_idx:
                p_belief = self.fleet.cars[viewer_idx]['belief_params'][k]
                vec.extend([p_belief['sf1'], p_belief['sf2'], p_belief['lookahead'], p_belief['speed'], p_belief['blocking']])
            
        X = torch.tensor([vec]).float().to(DEVICE)
        
        # --- Part C: Exact Normalization Logic Reconstruction ---
        # Calculate Relative Distances strictly matching original feature logic
        # Original: X[:,0] = s_opp2 - s_opp1; X[:,1] = s_opp1 - s_self; X[:,2] = s_opp2 - s_self
        
        s_self = X[0, 0].item() # Car 0 (Viewer)
        
        # We need to find indices in the tensor for s_self, s_opp1, s_opp2
        # S indices are 0, 1, 2...
        # If viewer_idx = 0: s_self=0, s_opp1=1, s_opp2=2
        # If viewer_idx = 1: s_self=1, s_opp1=2, s_opp2=0
        
        indices = [(viewer_idx + k) % self.fleet.num_cars for k in range(self.fleet.num_cars)]
        # indices[0] -> self, indices[1] -> opp1, indices[2] -> opp2
        
        # Helper to wrap
        def wrap_tensor(val):
            val = torch.where(val > 75., val - 150.087, val)
            val = torch.where(val < -75., val + 150.087, val)
            return val

        # Overwrite Tensor Values with Relative Distances
        # Note: We overwrite the S slots (0, 1, 2) with the specific relative features expected by the model
        X[:, :self.fleet.num_cars] = wrap_tensor(X[:, :self.fleet.num_cars] - s_self)
        if self.fleet.num_cars >= 3:
            # (Opp2 - Self) - (Opp1 - Self) = Opp2 - Opp1
            s_opp1_rel = X[:, 1]
            s_opp2_rel = X[:, 2]
            
            X[:, 0] = wrap_tensor(s_opp2_rel - s_opp1_rel)
        else:
            # only one Opp
            X[:, 0] = 0.0
        return X

    def collect_data(self):
        data = []
        c0 = self.fleet.cars[0]
        data.extend([c0['px'], c0['py'], c0['psi']])
        for k in range(1, self.fleet.num_cars):
            ck = self.fleet.cars[k]
            data.extend([ck['px'], ck['py'], ck['psi']])
        
        # Log Ego Params + Opponent Real Params (Matches original logging buffer)
        p = c0['opt_params']
        data.extend([p['sf1'], p['sf2'], p['lookahead'], p['speed'], p['blocking']])
        
        for k in range(1, self.fleet.num_cars):
             p = self.fleet.cars[k]['opt_params']
             data.extend([p['sf1'], p['sf2'], p['lookahead'], p['speed'], p['blocking']])
        
        data.extend([c0['action'][0], c0['action'][1]])
        self.fleet.buffer.append(data)

    def handle_episode_end(self):
        TRACK_LENGTH = 150.087 
        final_scores = []
        
        for k in range(self.fleet.num_cars):
            s_val = self.fleet.cars[k]['s']
            if s_val < 50.0:
                s_val += TRACK_LENGTH
            final_scores.append(s_val)

        win_idx = np.argmax(final_scores)
        self.fleet.n_wins[win_idx] += 1
        print(f"Episode {self.fleet.ep_no} Done. Scores: {final_scores}. Winner: Car {win_idx}")

        self.fleet.dataset.append(np.array(self.fleet.buffer))
        self.fleet.buffer = []
        
        regrets = {f'regrets{k+1}': self.optimizer.regrets[k] for k in range(self.fleet.num_cars)}
        if self.config['argparse']['save_data']:
            self.fleet.save_data(self.exp_name, regrets)
        
        max_episodes = self.config['argparse'].get('max_episodes', 100)
        
        if self.fleet.ep_no >= max_episodes:
            print(f"Experiment Completed: Reached {max_episodes} episodes.")
            self.fleet.save_data(self.exp_name + "_final", regrets)
            rospy.signal_shutdown(f"Finished {max_episodes} episodes")
            sys.exit(0)

        self.fleet.reset_episode()
        self.i = 1

    def slow_timer_callback(self, event):
        self.fleet.publish_static_map()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    default_yaml = 'evaluate.yaml'
    parser.add_argument('--config', default=default_yaml, 
                        help='Name of the configuration file (default: evaluate.yaml)')
    
    args = parser.parse_args()
    config_dir = os.path.join(_GAME_RACER_ROOT, 'config')
    full_config_path = os.path.join(config_dir, args.config)
    print(f"Final config path: {full_config_path}")
    
    try:
        CarNode(full_config_path)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass