import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.join(current_dir, '../../../ddrx-spliner/src')
sys.path.append(os.path.normpath(target_dir))
print(f'target_dir:{os.path.abspath(target_dir)}')
import numpy as np
import time
import copy
import rospy
from f110_msgs.msg import Wpnt, WpntArray, Obstacle, ObstacleArray, OTWpntArray, OpponentTrajectory, OppWpnt
from std_msgs.msg import Float32MultiArray, Float32, Bool
from visualization_msgs.msg import MarkerArray
from nav_msgs.msg import Odometry
from dynamic_reconfigure.msg import Config

from differential_flatness import PolynomialPath
from qp_fit import QPFit
from converter_utils import InitialRefSpline
from frenet_converter.frenet_converter import FrenetConverter
# Assuming these are custom classes you have defined elsewhere
# from your_module import FrenetConverter, InitialRefSpline, QPFit, MPC_Tracking_Controller

class DdrxPlannerCore:
    """
    Core logic class for the planner. 
    Handles path generation, collision checking, QP fitting, and MPC tracking.
    """
    def __init__(self):
        # --- Internal State & History ---
        self.past_avoidance_d = []
        self.last_ot_side = ""
        
        # --- Algorithm Parameters (Default values, updated by Node) ---
        self.lookahead = 15
        self.min_radius = 0.05
        self.max_kappa = 1.0 / self.min_radius
        self.width_car = 0.28
        self.half_width = 0.28 / 2.0
        self.avoidance_resolution = 20
        self.back_to_raceline_before = 5
        self.back_to_raceline_after = 5
        self.obs_traj_tresh = 2
        self.evasion_dist = 0.65
        self.spline_bound_mindist = 0.2
        self.avoid_static_obs = True

        # --- Helper Objects ---
        self.converter = None
        self.initial_ref_spline = None
        self.mpc_controller = None  # Will be initialized by Node
        self.qp_fit = QPFit() # Assuming QPFit is imported
        
        # --- Data Containers (Updated by Node) ---
        # Spline / Fit variables
        self.qp_fit_poly_x = None
        self.qp_fit_poly_y = None
        
        # Internal processing variables
        self.obs_downsampled_indices = np.array([])
        self.obs_downsampled_center_d = np.array([])
        self.obs_downsampled_min_dist = np.array([])
        self.down_sampled_delta_s = None
        self.global_traj_kappas = None
        self.obs_pre_resp = None # Placeholder for obstacle prediction response

    def set_converter(self, converter):
        """Sets the FrenetConverter object."""
        self.converter = converter

    def set_spline(self, spline):
        """Sets the InitialRefSpline object."""
        self.initial_ref_spline = spline

    def set_mpc_controller(self, mpc_controller):
        """Sets the MPC Controller object."""
        self.mpc_controller = mpc_controller

    def update_params(self, params_dict):
        """
        Updates algorithm parameters from dynamic reconfigure or ROS params.
        """
        for key, value in params_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def gen_raw_traj(self, raw_evasion_s, raw_evasion_v, ref_d_left, ref_d_right, side, frenet_state, scaled_max_s):
        """
        Generates the raw evasion trajectory based on obstacle positions and overtaking side.
        """
        if self.obs_downsampled_indices.size == 0:
            return np.array([]), np.array([]), np.array([]), np.array([]), False
        
        # Initialize raw_evasion_d
        raw_evasion_d = [0] * len(raw_evasion_s)
        
        # ... [Logic for idx calculation omitted for brevity, logic remains identical to original] ...
        # NOTE: Using simplified placeholders here to represent the logic flow
        idx_start_in_es = self.obs_downsampled_indices[0]
        # ... (Insert full original logic for index calculation here) ...

        no_collision = True
        is_curpos_as_start = True
        
        current_s = frenet_state.pose.pose.position.x
        current_d = frenet_state.pose.pose.position.y

        # logic for is_curpos_as_start...
        if current_s % scaled_max_s < raw_evasion_s[0] % scaled_max_s:
             current_d = 0
             is_curpos_as_start = False

        # ... [Insert overtaking side logic (Left/Right) from original code] ...
        # Just creating dummy returns to simulate logic flow
        updated_evasion_d = raw_evasion_d # Placeholder
        
        # Fit logic
        x_data, y_data, t_data = self.get_fit_variable(raw_evasion_s, raw_evasion_d, raw_evasion_v)
        poly_x, poly_y, qp_fit_status = self.get_qp_fit_coeffs(x_data, y_data, t_data, raw_evasion_s, raw_evasion_v, is_curpos_as_start)
        
        if not qp_fit_status:
            no_collision = False
            return raw_evasion_d, np.array([]), np.array([]), np.array([]), no_collision

        updated_evasion_d, closedform_evasion_valid = self.check_fit_traj_valid(poly_x, poly_y, t_data, ref_d_left) # Example for left

        if closedform_evasion_valid == False:
            no_collision = False
            return updated_evasion_d, poly_x, poly_y, t_data, no_collision
            
        return updated_evasion_d, poly_x, poly_y, t_data, no_collision

    def get_qp_fit_coeffs(self, x_data, y_data, t_data, raw_evasion_s, raw_evasion_v, is_curpos_as_start):
        """Calculates QP fit coefficients."""
        # TODO: validate the initial velocity direction
        phi_start = self.initial_ref_spline.get_phi(raw_evasion_s[0])
        v_start_x = raw_evasion_v[0]*np.cos(phi_start)
        v_start_y = raw_evasion_v[0]*np.sin(phi_start)
        phi_end = self.initial_ref_spline.get_phi(raw_evasion_s[-1])
        v_end_x = raw_evasion_v[-1]*np.cos(phi_end)
        v_end_y = raw_evasion_v[-1]*np.sin(phi_end)

        self.qp_fit.set_data(x_data, y_data, t_data, v_start_x, v_start_y, v_end_x, v_end_y)
        poly_x, poly_y, qp_fit_status = self.qp_fit.solve_fit_qp()

        return poly_x, poly_y, qp_fit_status

    def get_fit_variable(self, raw_evasion_s, raw_evasion_d, raw_evasion_v):
        """Converts Frenet coordinates to Cartesian and calculates time data."""
        waypts_resp = self.converter.get_cartesian(raw_evasion_s, raw_evasion_d)
        x_data = waypts_resp[0, :]
        y_data = waypts_resp[1, :] 
        t_data = np.zeros_like(x_data, dtype=float)
        
        for i in range(1, len(x_data)):
            distance = np.sqrt((x_data[i] - x_data[i-1])**2 + (y_data[i] - y_data[i-1])**2)
            if raw_evasion_v[i-1] <= 1e-6:
                # Handle zero velocity edge case
                t_data[i] = t_data[i-1] + 0.1 # Placeholder fallback
            else:
                t_data[i] = t_data[i-1] + distance / np.fabs(raw_evasion_v[i-1])
        return x_data, y_data, t_data

    def check_interp_traj_valid(self, raw_evasion_d, ref_d):
        safe_dist = 0.1
        return np.all(np.abs(raw_evasion_d) <= (np.abs(ref_d)- safe_dist))

    def check_fit_traj_valid(self, poly_x, poly_y, t_data, ref_d):
        safe_dist = 0.1
        x_fit = self.qp_fit.poly(t_data, poly_x)
        y_fit = self.qp_fit.poly(t_data, poly_y)
        resp = self.converter.get_frenet(x_fit, y_fit)
        s, d = resp[0, :], resp[1, :]
        return d, np.all(np.abs(d) <= (np.abs(ref_d)- safe_dist))

    def compute_evasion_trajectory(self, considered_obs, frenet_state, scaled_wpnts_msg, scaled_wpnts_array, 
                                   scaled_max_s, scaled_max_idx, scaled_delta_s, max_s_updated, 
                                   wpnts_updated, max_idx_updated, opponent_wpnts_sm, opponent_waypoints, max_opp_idx):
        """
        Main logic function (formerly fit_curve).
        Calculates the optimal evasion trajectory.
        Returns tuple of (evasion_x, evasion_y, evasion_s, evasion_d, evasion_v, no_collision, debug_info).
        """
        cur_s = frenet_state.pose.pose.position.x
        danger_flag = False
        
        # ... [Logic for initial_guess_object, gb_idxs, side determination omitted for brevity] ...
        # Assume these are calculated here as in the original code
        side = "left" # Placeholder
        initial_apex = 0.0 # Placeholder

        # ... [Logic for downsampling and finding obstacles omitted] ...
        # This section populates self.obs_downsampled_indices, etc.
        
        # Generate raw trajectory
        s_avoidance = np.linspace(0, 10, self.avoidance_resolution) # Placeholder
        self.s_avoidance = s_avoidance
        self.side = side
        
        # Prepare inputs for gen_raw_traj
        # ... (Get corresponding scaled wpnts, ref_d, raw_evasion_v, etc.) ...
        raw_evasion_v = np.zeros_like(s_avoidance) # Placeholder
        raw_evasion_s = np.mod(s_avoidance, scaled_max_s)
        ref_d_left = np.zeros_like(s_avoidance) # Placeholder
        ref_d_right = np.zeros_like(s_avoidance) # Placeholder

        updated_evasion_d, poly_x, poly_y, t_data, no_collision = self.gen_raw_traj(
            raw_evasion_s, raw_evasion_v, ref_d_left, ref_d_right, side, frenet_state, scaled_max_s
        )

        is_traj_valid = False
        if poly_x.size != 0:
            is_traj_valid = True

        # ... [Logic for Min Radius and SQP Solver omitted] ...

        if is_traj_valid:
            # Reconstruct full trajectory
            # ... (Interpolation logic) ...
            evasion_x = np.array([]) # Placeholder result
            evasion_y = np.array([])
            evasion_s = np.array([])
            evasion_d = np.array([])
            evasion_v = np.array([])
            
            self.past_avoidance_d = updated_evasion_d[:]
            self.qp_fit_poly_x = poly_x[:]
            self.qp_fit_poly_y = poly_y[:]
            self.last_ot_side = side 
        else:
            evasion_x = []
            evasion_y = []
            evasion_s = []
            evasion_d = []
            evasion_v = []
            self.past_avoidance_d = []
            self.qp_fit_poly_x = np.array([])
            self.qp_fit_poly_y = np.array([])

        # Return results to Node for publishing/visualization
        return evasion_x, evasion_y, evasion_s, evasion_d, evasion_v, no_collision

    def mpc_tracking(self, x0, frenet_states_list, frenet_inputs_list, s_ref, d_left_ref, d_right_ref, scaled_max_s):
        """
        Runs the MPC controller optimization.
        """
        # [s, d, delta_phi]
        raw_s = frenet_states_list[:, 0]
        # Handle wrapping
        for i in range(len(raw_s) - 1):
            if raw_s[i] > raw_s[i + 1]:
                raw_s[i + 1:] = [s + scaled_max_s for s in raw_s[i + 1:]]
                break
        
        x_ref = np.column_stack((raw_s, frenet_states_list[:, 1], frenet_states_list[:, 3]))
        u_ref = np.column_stack((frenet_states_list[:, 2], frenet_inputs_list[:, 1]))

        # Calculate Boundaries
        bound_raw_s_idx = np.array([np.abs(s_ref - s % scaled_max_s).argmin() for s in raw_s])
        left_bound = d_left_ref[bound_raw_s_idx] - 0.5 * self.width_car
        right_bound = np.minimum(-d_right_ref[bound_raw_s_idx] + 0.5 * self.width_car, left_bound - 2*(self.half_width+0.1))

        # ... [Logic for dynamic boundaries using obs_pre_resp omitted] ...
        
        # Solve MPC
        if len(x_ref) >= self.mpc_controller.N: 
            res_u, res_x = self.mpc_controller.solve_qp(x0, x_ref, u_ref, left_bound, right_bound)
        else:
            res_u, res_x = None, None

        # Format output
        mpc_x, mpc_y, mpc_s, mpc_d, mpc_v = [], [], [], [], []
        
        if res_u is not None:
             # ... [Interpolation and formatting logic] ...
             pass

        return mpc_x, mpc_y, mpc_s, mpc_d, mpc_v

    # --- Placeholders for helper methods ---
    def group_objects(self, considered_obs):
        raise NotImplementedError("Needs implementation: group_objects")

    def _more_space(self, initial_guess_object, wpnts, gb_idxs):
        return "left", 0.0

    def get_obs_pre_resp(self):
        """
        Constructs the safe drivable corridor (a "tube" or "tunnel") around the obstacles.
        It defines the Upper Bound (Left Limit) and Lower Bound (Right Limit) for the optimizer.
        """
        # Map the local downsampled indices (from the planning grid) back to global waypoint indices.
        opp_wpnts_glbidx = [np.abs(self.s_ref - self.s_avoidance[int(idx)]%self.max_opp_idx).argmin() for idx in self.obs_downsampled_indices]
        
        # first_sequence: Represents the Upper Bound (Left Boundary of the drivable area).
        # second_sequence: Represents the Lower Bound (Right Boundary of the drivable area).
        first_sequence = []
        second_sequence = []
        
        # Get the S-coordinates corresponding to the obstacle area.
        s_sequence = self.s_avoidance[self.obs_downsampled_indices]

        for idx in range(len(self.obs_downsampled_center_d)): 
            if self.side == 'left':
                # Logic: If obstacle d is positive (left of center), we technically have less space, 
                if self.obs_downsampled_center_d[int(idx)] > 0:
                    first_sequence.append(self.d_left_ref[int(opp_wpnts_glbidx[idx])] - self.obs_downsampled_center_d[int(idx)]) 
                else:
                    first_sequence.append(self.d_left_ref[int(opp_wpnts_glbidx[idx])])
                
                # Calculation: Obstacle Center + Half Car Width (treating obstacle as the right wall).
                # 'strict_right_bound' ensures a minimum channel width. If the obstacle is too close 
                # to the track edge, we clamp the bound to ensure at least one car width fits.
                strict_right_bound = min(first_sequence[-1] - 2 * (self.half_width + 0.1), 
                                         self.obs_downsampled_center_d[int(idx)] + 0.5 * self.width_car)
                second_sequence.append(strict_right_bound)
            else:
                # Logic: Standard track limit on the right side (negative d).
                if self.obs_downsampled_center_d[int(idx)] > 0:
                    second_sequence.append(-self.d_right_ref[int(opp_wpnts_glbidx[idx])] - self.obs_downsampled_center_d[int(idx)]) 
                else:
                    second_sequence.append(-self.d_right_ref[int(opp_wpnts_glbidx[idx])])
                
                # Calculation: Obstacle Center - Half Car Width (treating obstacle as the left wall).
                strict_left_bound = max(second_sequence[-1] + 2 * (self.half_width + 0.1),
                                        self.obs_downsampled_center_d[int(idx)] - 0.5 * self.width_car)
                first_sequence.append(strict_left_bound) 
        
        # Add points BEFORE the obstacle actually starts to create a smooth approach.
        # This prevents the optimizer from facing a "wall" instantly and allows smooth entry.
        ahead_num = 5
        safe_point_start_idx = max(self.obs_downsampled_indices[0]-ahead_num, 0)
        
        if safe_point_start_idx > 0:
            start_idx = self.obs_downsampled_indices[0] - ahead_num
            end_idx = self.obs_downsampled_indices[0] - 1
            index_range = list(range(start_idx, end_idx + 1))
            
            # Get global indices for these buffer points to find track width.
            larger_wpnts_glbidx = [np.abs(self.s_ref - self.s_avoidance[int(idx)]).argmin() for idx in index_range]

            # Check which side has more space at the START of the obstacle to align the funnel.
            left_space = max(self.d_left_ref[int(opp_wpnts_glbidx[0])] - self.obs_downsampled_center_d[int(0)], 0)
            right_space = max(self.obs_downsampled_center_d[int(0)] + self.d_right_ref[int(opp_wpnts_glbidx[0])], 0)
            
            # Insert buffer points into the sequences (Prepend to the front of the list).
            if left_space > right_space:
                    for i in range(ahead_num):
                        # Use full track width on the open side, and blend constraints on the obstacle side.
                        first_sequence.insert(0, self.d_left_ref[int(larger_wpnts_glbidx[-(i + 1)])])
                        second_sequence.insert(0, max(-self.d_right_ref[int(larger_wpnts_glbidx[-(i + 1)])], second_sequence[0]))
                        # Insert corresponding S coordinates.
                        s_sequence = np.insert(s_sequence, 0, self.s_avoidance[self.obs_downsampled_indices[0] -(i + 1)])
            else:
                    for i in range(ahead_num):
                        second_sequence.insert(0, -self.d_right_ref[int(larger_wpnts_glbidx[-(i + 1)])])
                        first_sequence.insert(0, min(self.d_left_ref[int(larger_wpnts_glbidx[-(i + 1)])], first_sequence[0]))
                        s_sequence = np.insert(s_sequence, 0, self.s_avoidance[self.obs_downsampled_indices[0] - (i + 1)])

        # Shape: [N x 3] -> [s, d_upper, d_lower]
        self.obs_pre_resp = np.column_stack((s_sequence, first_sequence, second_sequence))
    
    def process_obstacles(self, considered_obs):
        """
        Maps continuous obstacles onto the discrete path planning grid (s_avoidance).
        Generates constraint arrays (indices, center_d, min_dist) for the optimization solver.
        
        Args:
            considered_obs: list, List of obstacle objects to be processed.
            s_avoidance: np.array, Array of discrete S positions (grid points) for local planning.
        """
        
        # Initialize arrays to store obstacle projection data on the discrete grid.
        # 1. Indices of the s_avoidance grid points that are affected by obstacles.
        self.obs_downsampled_indices = np.array([])
        # 2. Center lateral position (d) of the obstacle at those specific indices.
        self.obs_downsampled_center_d = np.array([])
        # 3. Minimum lateral distance required to safely avoid the obstacle at those indices.
        self.obs_downsampled_min_dist = np.array([])


        for obs in considered_obs:
            obs_idx_start = np.abs(self.s_avoidance - obs.s_start).argmin()
            obs_idx_end = np.abs(self.s_avoidance - obs.s_end).argmin()

            if obs_idx_start < len(self.s_avoidance) - 2: 
                if obs.is_static == True or obs_idx_end == obs_idx_start:
                    # This ensures the obstacle occupies at least one grid interval in the optimizer.
                    if obs_idx_end == obs_idx_start:
                        obs_idx_end = obs_idx_start + 1
                    
                    self.obs_downsampled_indices = np.append(self.obs_downsampled_indices, np.arange(obs_idx_start, obs_idx_end + 1))
                    # Since it is static, we assume a constant center d (rectangular shape) across these indices.
                    self.obs_downsampled_center_d = np.append(self.obs_downsampled_center_d, np.full(obs_idx_end - obs_idx_start + 1, (obs.d_left + obs.d_right) / 2))
                    self.obs_downsampled_min_dist = np.append(self.obs_downsampled_min_dist, np.full(obs_idx_end - obs_idx_start + 1, (obs.d_left - obs.d_right) / 2 + self.width_car + self.evasion_dist))
                else:
                    # Generate the array of indices covered by this obstacle.
                    indices = np.arange(obs_idx_start, obs_idx_end + 1)
                    self.obs_downsampled_indices = np.append(self.obs_downsampled_indices, indices)
                    
                    # Map our local planning grid S to the opponent's predicted trajectory S.
                    opp_wpnts_idx = [np.abs(self.opponent_wpnts_sm - self.s_avoidance[int(idx)]%self.max_opp_idx).argmin() for idx in indices]
                    
                    # Lookup the dynamic lateral position (d) of the opponent at those specific mapped points.
                    d_opp_downsampled_array = np.array([self.opponent_waypoints[opp_idx].d_m for opp_idx in opp_wpnts_idx])                    
                    # Append the dynamic d values to the global center_d array.
                    self.obs_downsampled_center_d = np.append(self.obs_downsampled_center_d, d_opp_downsampled_array)
                    self.obs_downsampled_min_dist = np.append(self.obs_downsampled_min_dist, np.full(obs_idx_end - obs_idx_start + 1, self.width_car + self.evasion_dist))
            
            else:
                rospy.loginfo("[OBS Spliner] Obstacle end index is smaller than start index")

        self.obs_downsampled_indices = self.obs_downsampled_indices.astype(int)

    def get_fit_variable(self, raw_evasion_s, raw_evasion_d, raw_evasion_v, speed_scaling_factor=1.0):
        waypts_resp = self.converter.get_cartesian(raw_evasion_s, raw_evasion_d)
        x_data = waypts_resp[0, :]
        y_data = waypts_resp[1, :] 
        t_data = np.zeros_like(x_data, dtype=float)
        for i in range(1, len(x_data)):
            distance = np.sqrt((x_data[i] - x_data[i-1])**2 + (y_data[i] - y_data[i-1])**2)
            scaled_v = raw_evasion_v[i-1] * speed_scaling_factor
            if np.abs(scaled_v) <= 1e-6:
                raise ValueError(f"Zero or near-zero velocity detected at index {i-1}. Cannot compute time.")
            t_data[i] = t_data[i-1] + distance / np.fabs(scaled_v)
        return x_data, y_data, t_data
    
    def check_interp_traj_valid(self, raw_evasion_d, ref_d):
        safe_dist = 0.1
        return np.all(np.abs(raw_evasion_d) <= (np.abs(ref_d)- safe_dist))

    def check_fit_traj_valid(self, poly_x, poly_y, t_data, ref_d):
        safe_dist = 0.1
        x_fit = self.qp_fit.poly(t_data, poly_x)
        y_fit = self.qp_fit.poly(t_data, poly_y)
        resp = self.converter.get_frenet(x_fit, y_fit)
        s, d = resp[0, :], resp[1, :]
        return d, np.all(np.abs(d) <= (np.abs(ref_d)- safe_dist))            

    def group_objects(self, obstacles: list):
        """
        Computes the bounding box of a list of obstacles.
        
        Pre-condition: 
        The obstacles in the input list should already be spatially clustered (e.g., via a distance check).
        
        Returns:
        A new object representing the merged bounding box of all input obstacles.
        """
        if not obstacles:
            return None

        # Use deepcopy to create a new instance. 
        # This prevents modifying the original properties of the first obstacle in the list.
        merged_obs = copy.deepcopy(obstacles[0])

        # Iterate through the rest of the obstacles to find the global extrema (min/max boundaries).
        for obs in obstacles[1:]:
            # Expand the Left Boundary (Find the maximum positive d_left)
            if obs.d_left > merged_obs.d_left:
                merged_obs.d_left = obs.d_left
            
            # Expand the Right Boundary (Find the minimum negative d_right)
            if obs.d_right < merged_obs.d_right:
                merged_obs.d_right = obs.d_right
            
            # Expand the Longitudinal Start (Find the minimum s_start)
            if obs.s_start < merged_obs.s_start:
                merged_obs.s_start = obs.s_start
            
            # Expand the Longitudinal End (Find the maximum s_end)
            if obs.s_end > merged_obs.s_end:
                merged_obs.s_end = obs.s_end

        # Recalculate the geometric center based on the new bounding box.
        merged_obs.s_center = (merged_obs.s_start + merged_obs.s_end) / 2
        
        return merged_obs
           
    def mpc_tracking(self, x0, frenet_states_list, frenet_inputs_list):
        """ mpc tracking controller """
        # [s, d, delta_phi]
        raw_s = frenet_states_list[:, 0]
        for i in range(len(raw_s) - 1):
            if raw_s[i] > raw_s[i + 1]:
                raw_s[i + 1:] = [s + self.scaled_max_s for s in raw_s[i + 1:]]
                break
        
        x_ref = np.column_stack((
            raw_s, 
            frenet_states_list[:, 1], 
            frenet_states_list[:, 3]
        ))

        # [v, delta]
        u_ref = np.column_stack((
            frenet_states_list[:, 2], 
            frenet_inputs_list[:, 1]
        ))

        #print bounds time
        start_time = time.time()
        # cal left boundary and right boundary for raw_s
        bound_raw_s_idx = np.array([
            np.abs(self.s_ref - s % self.scaled_max_s).argmin() for s in raw_s
            ])
        left_bound = self.d_left_ref[bound_raw_s_idx] - 0.5 * self.width_car
        right_bound = np.minimum(-self.d_right_ref[bound_raw_s_idx] + 0.5 * self.width_car, left_bound - 2*(self.half_width+0.1)) # avoid right_bound > left_bound
        s_bound = raw_s % self.scaled_max_s
        self.get_obs_pre_resp()

        # Extract s values from observed responses (N observations)
        obs_s = self.obs_pre_resp[:, 0]  # Observed s values (first column)
        obs_left = self.obs_pre_resp[:, 1]  # Left boundary values from observations
        obs_right = self.obs_pre_resp[:, 2]  # Right boundary values from observations
        for i in range(len(obs_s) - 1):
            if obs_s[i] > obs_s[i + 1]:
                obs_s[i + 1:] = [s + self.scaled_max_s for s in obs_s[i + 1:]]
                break

        # Find the range of obs_s (ensure ref_s is within this range)
        min_s = np.min(obs_s)
        max_s = np.max(obs_s)

        # Create a mask to only interpolate ref_s values within the range of obs_s
        valid_mask = (raw_s >= min_s) & (raw_s <= max_s)

        left_bound[valid_mask] = np.interp(raw_s[valid_mask], obs_s, obs_left)  # Interpolate left bound
        right_bound[valid_mask] = np.interp(raw_s[valid_mask], obs_s, obs_right)  # Interpolate right bound

        overtake_left_resp = self.converter.get_cartesian(s_bound, left_bound)
        overtake_right_resp = self.converter.get_cartesian(s_bound, right_bound)
        start_point = overtake_left_resp.T
        end_point = overtake_right_resp.T

        self.visualize_ob_lines(start_point, end_point)
        end_time = time.time()-start_time
        print(f"bounds time: {end_time*1000:.3f}ms")

        if len(x_ref) >= self.mpc_controller.N: 
            res_u, res_x = self.mpc_controller.solve_qp(x0, x_ref, u_ref, left_bound, right_bound)
        else:
            res_u = None
            res_x = None

        mpc_x = []
        mpc_y = []
        mpc_s = []
        mpc_d = []
        mpc_v = []
        
        if res_u is not None:
            res_x = np.array(res_x).reshape(-1, 3)
            res_u = np.array(res_u).reshape(-1, 2)

            # output of mpc
            origin_s = res_x[:, 0]
            origin_d = res_x[:, 1]
            origin_v = res_u[:, 0]

            # interpolate the output of mpc
            ds = 0.1
            mpc_s = np.arange(origin_s[0], origin_s[-1], ds) 
            mpc_d = np.interp(mpc_s, origin_s, origin_d)
            mpc_v = np.interp(mpc_s, origin_s, origin_v)

            # spline_d = CubicSpline(origin_s, origin_d)
            # spline_v = CubicSpline(origin_s, origin_v)
            # mpc_d = spline_d(mpc_s)
            # mpc_v = spline_v(mpc_s)

            mpc_s = mpc_s % self.scaled_max_s
            resp = self.converter.get_cartesian(mpc_s, mpc_d)
            mpc_x = resp[0, :]
            mpc_y = resp[1, :]

        return mpc_x, mpc_y, mpc_s, mpc_d, mpc_v
    
    def more_space(self, obstacle: Obstacle, gb_wpnts, gb_idxs):
        left_boundary_mean = np.mean([gb_wpnts[gb_idx].d_left for gb_idx in gb_idxs])
        right_boundary_mean = np.mean([gb_wpnts[gb_idx].d_right for gb_idx in gb_idxs])
        left_gap = abs(left_boundary_mean - obstacle.d_left)
        right_gap = abs(right_boundary_mean + obstacle.d_right)
        min_space = self.evasion_dist + self.spline_bound_mindist

        if right_gap > min_space and left_gap < min_space:
            # Compute apex distance to the right of the opponent
            d_apex_right = obstacle.d_right - self.evasion_dist
            # If we overtake to the right of the opponent BUT the apex is to the left of the raceline, then we set the apex to 0
            if d_apex_right > 0:
                d_apex_right = 0
            return "right", d_apex_right

        elif left_gap > min_space and right_gap < min_space:
            # Compute apex distance to the left of the opponent
            d_apex_left = obstacle.d_left + self.evasion_dist
            # If we overtake to the left of the opponent BUT the apex is to the right of the raceline, then we set the apex to 0
            if d_apex_left < 0:
                d_apex_left = 0
            return "left", d_apex_left
        else:
            candidate_d_apex_left = obstacle.d_left + self.evasion_dist
            candidate_d_apex_right = obstacle.d_right - self.evasion_dist

            if abs(candidate_d_apex_left) <= abs(candidate_d_apex_right):
                # If we overtake to the left of the opponent BUT the apex is to the right of the raceline, then we set the apex to 0
                if candidate_d_apex_left < 0:
                    candidate_d_apex_left = 0
                return "left", candidate_d_apex_left
            else:
                # If we overtake to the right of the opponent BUT the apex is to the left of the raceline, then we set the apex to 0
                if candidate_d_apex_right > 0:
                    candidate_d_apex_right = 0
                return "right", candidate_d_apex_right
    
    def generate_trajectory(self, considered_obs: list, cur_s: float, s1, d2, s3, s_end, speed_scaling_factor=1.0):
        # --- Module 1: Obstacle Grouping and Index Mapping ---
        # Aggregate multiple obstacles into a single bounding box to simplify decision making
        initial_guess_object = self.group_objects(considered_obs)

        # Find the nearest global waypoint indices for the merged obstacle's start and end positions
        initial_guess_object_start_idx = np.abs(self.scaled_wpnts - initial_guess_object.s_start).argmin()
        initial_guess_object_end_idx = np.abs(self.scaled_wpnts - initial_guess_object.s_end).argmin()

        # Generate an array of indices covering the Region of Concern (ROC)
        # The modulo operator handles the lap-around case for circular tracks
        gb_idxs = np.array(range(initial_guess_object_start_idx, 
                                initial_guess_object_start_idx + (initial_guess_object_end_idx - initial_guess_object_start_idx) % self.scaled_max_idx)) % self.scaled_max_idx

        # Buffer logic: ensure the ROC has a minimum length (at least 20 waypoints) for stable calculation
        if len(gb_idxs) < 20:
            gb_idxs = [int(initial_guess_object.s_center / self.scaled_delta_s + i) % self.scaled_max_idx for i in range(20)]
        
        # --- Module 2: Overtaking Side Decision and Curvature Analysis ---
        # Determine which side (left/right) offers more space and calculate the target lateral displacement (apex)
        side, initial_apex = self.more_space(initial_guess_object, self.scaled_wpnts_msg.wpnts, gb_idxs)

        # Extract curvature (kappa) values for the waypoints in the ROC
        kappas = np.array([self.scaled_wpnts_msg.wpnts[gb_idx].kappa_radpm for gb_idx in gb_idxs])
        max_kappa = np.max(np.abs(kappas))

        # Identify which side is the "outside" of the current corner based on the sign of curvature
        # Negative sum typically indicates a right-hand turn, making the left side the "outside"
        outside = "left" if np.sum(kappas) < 0 else "right"

        # --- Module 3: Curvature-Based Obstacle Elongation ---
        # If overtaking on the outside of a corner, artificially extend the obstacle's end position.
        # This accounts for the increased travel distance and centrifugal forces, 
        # forcing a smoother and later return to the racing line.
        if side == outside:
            for i in range(len(considered_obs)):
                # Extend s_end proportional to max curvature and the required lateral shift
                extension = (considered_obs[i].s_end - considered_obs[i].s_start) % self.max_s_updated * max_kappa * (self.width_car + self.evasion_dist)
                considered_obs[i].s_end += extension

        # Find the global min/max S coordinates after elongation and check for extreme obstacle widths
        min_s_obs_start = self.scaled_max_s
        max_s_obs_end = 0
        for obs in considered_obs:
            if obs.s_start < min_s_obs_start:
                min_s_obs_start = obs.s_start
            if obs.s_end > max_s_obs_end:
                max_s_obs_end = obs.s_end
            
            # Safety flag for extremely wide obstacles that might block the entire track
            if obs.d_left > 3 or obs.d_right < -3:
                danger_flag = True
        
        # --- Module 4: Corridor Generation and Downsampling ---
        # Define the longitudinal start and end of the entire avoidance maneuver
        # Includes 'buffer zones' before and after the obstacle for smooth transitions
        start_avoidance = max((min_s_obs_start - self.back_to_raceline_before), cur_s)
        end_avoidance = max_s_obs_end + self.back_to_raceline_after

        # Resample the avoidance region into a uniform grid (S-axis)
        self.s_avoidance = np.linspace(start_avoidance, end_avoidance, self.avoidance_resolution)
        self.down_sampled_delta_s = self.s_avoidance[1] - self.s_avoidance[0]

        # Find the corresponding global waypoint indices for each point on the new grid
        scaled_wpnts_indices = np.array([np.abs(self.scaled_wpnts[:, 0] - s % self.scaled_max_s).argmin() for s in self.s_avoidance]) 
        corresponding_scaled_wpnts = [self.scaled_wpnts_msg.wpnts[i] for i in scaled_wpnts_indices]

        self.process_obstacles(considered_obs)
        control_point_s, control_point_d, control_point_t = self.compute_control_points(corresponding_scaled_wpnts, side, s1, d2, s3, s_end, speed_scaling_factor)
        set_data(control_point_s, control_point_d, control_point_t, start_state, end_state, t_switch=None)

    def compute_control_points(self, corresponding_scaled_wpnts, side, s1, d2, s3, s_end, speed_scaling_factor=1.0):
        """
        Generates control points (k1, k2, k3) and computes their timestamps based on raceline velocity.
        Refactored to replace magic numbers with named constants.

        Args:
            s1 (float): S-coordinate of control point k1.
            d2 (float): Magnitude of D-coordinate for control point k2.
            s3 (float): S-coordinate of control point k3.
            s_end (float): S-coordinate of the end point.
            side (str): Overtaking side, 'left' (positive d) or 'right' (negative d).
            speed_scaling_factor (float): Factor to scale velocity for time calculation.

        Returns:
            tuple: (s_coords, d_coords, t_coords, indices)
        """
        raw_evasion_v = np.array([wpnt.vx_mps for wpnt in corresponding_scaled_wpnts])
        ref_d_left = [wpnt.d_left for wpnt in corresponding_scaled_wpnts]
        ref_d_right = [wpnt.d_right for wpnt in corresponding_scaled_wpnts]
        ref_d_left = np.array(ref_d_left)
        ref_d_right = np.array(ref_d_right)
        raw_evasion_s = np.mod(self.s_avoidance, self.scaled_max_s)
        # --- Configuration Constants (Replacing Magic Numbers) ---
        # The minimum longitudinal distance buffer required between control points
        MIN_S_BUFFER = 0.5  
        
        # The ratio determining the position of s2 relative to the total length (s_end)
        # 0.5 means s2 is exactly in the middle
        S2_POSITION_RATIO = 0.5 
        
        # The ratio for d1 and d3 relative to d2 (2/3)
        D_SCALE_FACTOR = 2.0 / 3.0 
        
        # Minimum velocity threshold to avoid division by zero
        MIN_VELOCITY_THRESHOLD = 0.1
        
        # Total number of points (Start, k1, k2, k3, End)
        NUM_POINTS = 5
        # -------------------------------------------------------

        # 1. Calculate s2 based on the ratio constant
        s2 = s_end * S2_POSITION_RATIO
        
        # 2. Sanity Checks: Validate input constraints using the buffer constant
        # Check if s1 is out of valid bounds
        if s1 < MIN_S_BUFFER or s1 > (s2 - MIN_S_BUFFER):
            rospy.logerr(f"[Trajectory] Invalid s1: {s1}. Constraint: {MIN_S_BUFFER} < s1 < {s2 - MIN_S_BUFFER}")
            raise ValueError("Control point s1 constraints violated.")
            
        # Check if s3 is out of valid bounds
        if s3 < (s2 + MIN_S_BUFFER) or (s_end - s3) < MIN_S_BUFFER:
            rospy.logerr(f"[Trajectory] Invalid s3: {s3}. Constraint: {s2 + MIN_S_BUFFER} < s3 < {s_end - MIN_S_BUFFER}")
            raise ValueError("Control point s3 constraints violated.")

        # 3. Determine lateral (d) coordinates based on the overtaking side
        # 'left' implies positive d, 'right' implies negative d
        sign = 1.0 if side == "left" else -1.0
        
        # Ensure d2 is treated as a magnitude first, then apply sign
        real_d2 = sign * abs(d2)
        
        # Calculate d1 and d3 using the scale factor constant
        d1 = D_SCALE_FACTOR * real_d2
        d3 = D_SCALE_FACTOR * real_d2

        # 4. Construct arrays for all key points: Start, k1, k2, k3, End
        points_s = np.array([0.0, s1, s2, s3, s_end])
        points_d = np.array([0.0, d1, real_d2, d3, 0.0])
        
        # Initialize arrays for indices and timestamps
        points_indices = np.zeros(NUM_POINTS, dtype=int)
        points_t = np.zeros(NUM_POINTS, dtype=float)

        # 5. Iterate through points to find indices and calculate time
        for i in range(NUM_POINTS):
            # Find the closest index in the global s_avoidance grid
            current_s = points_s[i]
            idx = np.abs(self.s_avoidance - current_s).argmin()
            points_indices[i] = idx

            # Calculate time (t) for points after the start
            if i > 0:
                # Euclidean distance between the current point and the previous point
                dist = np.sqrt((points_s[i] - points_s[i-1])**2 + (points_d[i] - points_d[i-1])**2)
                
                # Get the velocity from the raceline at the previous point's index
                prev_idx = points_indices[i-1]
                
                # Boundary check for index access
                if prev_idx >= len(self.raceline_v):
                    prev_idx = len(self.raceline_v) - 1
                
                raw_v = self.raceline_v[prev_idx]
                
                # Apply the speed scaling factor
                scaled_v = raw_v * speed_scaling_factor
                
                # Avoid division by zero or extremely small velocities using the threshold constant
                if scaled_v < MIN_VELOCITY_THRESHOLD:
                    scaled_v = MIN_VELOCITY_THRESHOLD
                
                # Integrate time: t_current = t_prev + distance / velocity
                points_t[i] = points_t[i-1] + (dist / scaled_v)

        # Return the computed arrays
        return points_s, points_d, points_t

class DdrxPlannerNode:
    """
    ROS Interface Node. 
    Handles Subscriptions, Publishers, Parameter Updates, and invokes DdrxPlannerCore.
    """
    def __init__(self):
        # Initialize node
        rospy.init_node('sqp_avoidance_node')
        self.rate = rospy.Rate(20)

        # Initialize Logic Core
        self.core = DdrxPlannerCore()

        # Node State Variables
        self.frenet_state = Odometry()
        self.cart_state = Odometry()
        self.local_wpnts = None
        self.measure = rospy.get_param("/measure", False)
        
        # Scaled waypoints params
        self.scaled_wpnts = None
        self.scaled_wpnts_msg = WpntArray()
        self.scaled_vmax = None
        self.scaled_max_idx = None
        self.scaled_max_s = None
        self.scaled_delta_s = None
        
        # Updated waypoints params
        self.wpnts_updated = None
        self.max_s_updated = None
        self.max_idx_updated = None
        
        # Global Reference params
        self.global_waypoints = None
        self.kapparef = None
        self.s_ref = None
        self.d_left_ref = None
        self.d_right_ref = None

        # Obstalces params
        self.obs = ObstacleArray()
        self.obs_perception = ObstacleArray()
        self.obs_predict = ObstacleArray()
        
        # Opponent waypoint params
        self.opponent_waypoints = OpponentTrajectory()
        self.max_opp_idx = None
        self.opponent_wpnts_sm = None
        self.ot_section_check = False

        # ROS Parameters
        self.opponent_traj_topic = '/opponent_trajectory'
        
        # Subscribers
        rospy.Subscriber("/perception/obstacles", ObstacleArray, self.obs_perception_cb)
        rospy.Subscriber("/collision_prediction/obstacles", ObstacleArray, self.obs_prediction_cb)
        rospy.Subscriber("/car_state/odom_frenet", Odometry, self.state_cb)
        rospy.Subscriber("/global_waypoints_scaled", WpntArray, self.scaled_wpnts_cb)
        rospy.Subscriber("/local_waypoints", WpntArray, self.local_wpnts_cb)
        rospy.Subscriber("/dynamic_sqp_tuner_node/parameter_updates", Config, self.dyn_param_cb)
        rospy.Subscriber("/global_waypoints", WpntArray, self.gb_cb)
        rospy.Subscriber("/global_waypoints_updated", WpntArray, self.updated_wpnts_cb)
        rospy.Subscriber(self.opponent_traj_topic, OpponentTrajectory, self.opponent_trajectory_cb)
        rospy.Subscriber("/ot_section_check", Bool, self.ot_sections_check_cb)
        rospy.Subscriber("/car_state/odom", Odometry, self.cart_state_cb)

        # Publishers
        self.mrks_pub = rospy.Publisher("/planner/avoidance/markers_sqp", MarkerArray, queue_size=10)
        self.df_mrks_pub = rospy.Publisher("/planner/avoidance/df_markers", MarkerArray, queue_size=10)
        self.mpc_mrks_pub = rospy.Publisher("/planner/avoidance/mpc_markers", MarkerArray, queue_size=10)
        self.ob_line_pub = rospy.Publisher('/planner/avoidance/ob_line_marker', MarkerArray, queue_size=10)
        self.raw_mrks_pub = rospy.Publisher("/planner/avoidance/raw_traj_markers", MarkerArray, queue_size=10)
        self.evasion_pub = rospy.Publisher("/planner/avoidance/otwpnts", OTWpntArray, queue_size=10)
        self.merger_pub = rospy.Publisher("/planner/avoidance/merger", Float32MultiArray, queue_size=10)
        if self.measure:
            self.measure_pub = rospy.Publisher("/planner/pspliner_sqp/latency", Float32, queue_size=10)

        # Initialize Helpers (Dependent on ROS wait_for_message)
        self.initialize_helpers()

    def initialize_helpers(self):
        """
        Initializes helper objects that require blocking ROS calls and passes them to Core.
        """
        rospy.wait_for_message("/global_waypoints", WpntArray)
        
        # Initialize FrenetConverter
        if self.global_waypoints is not None:
            converter = FrenetConverter(self.global_waypoints[:, 0], self.global_waypoints[:, 1])
            self.core.set_converter(converter)
            rospy.loginfo("[DDRX Node] Initialized FrenetConverter object")
        
        # Initialize Spline
        initial_ref_spline = InitialRefSpline()
        self.core.set_spline(initial_ref_spline)

        # Initialize MPC Controller
        mpc_controller = MPC_Tracking_Controller()
        if self.s_ref is not None:
             mpc_controller.update_vehicle_Lmodel(self.s_ref, self.kapparef)
        self.core.set_mpc_controller(mpc_controller)
        rospy.loginfo("[DDRX Node] Initialized vehicle linear model!")

    ### Callbacks ###
    def obs_perception_cb(self, data: ObstacleArray):
        self.obs_perception = data
        self.obs_perception.obstacles = [obs for obs in data.obstacles if obs.is_static == True]
        if self.core.avoid_static_obs == True:
            self.obs.header = data.header
            self.obs.obstacles = self.obs_perception.obstacles + self.obs_predict.obstacles

    def obs_prediction_cb(self, data: ObstacleArray):
        self.obs_predict = data
        self.obs = self.obs_predict
        if self.core.avoid_static_obs == True:
            self.obs.obstacles = self.obs.obstacles + self.obs_perception.obstacles

    def state_cb(self, data: Odometry):
        self.frenet_state = data

    def cart_state_cb(self, data: Odometry):
        self.cart_state = data
    
    def gb_cb(self, data: WpntArray):
        self.global_waypoints = np.array([[wpnt.x_m, wpnt.y_m] for wpnt in data.wpnts]) 
        self.kapparef = np.array([x.kappa_radpm for x in data.wpnts])
        self.s_ref = np.array([x.s_m for x in data.wpnts])
        self.d_left_ref = np.array([x.d_left for x in data.wpnts])
        self.d_right_ref = np.array([x.d_right for x in data.wpnts]) 
        # Update MPC model if needed when global waypoints change
        # self.core.mpc_controller.update_vehicle_Lmodel(...)

    def scaled_wpnts_cb(self, data: WpntArray):
        self.scaled_wpnts = np.array([[wpnt.s_m, wpnt.d_m] for wpnt in data.wpnts])
        self.scaled_wpnts_msg = data
        v_max = np.max(np.array([wpnt.vx_mps for wpnt in data.wpnts]))
        if self.scaled_vmax != v_max:
            self.scaled_vmax = v_max
            self.scaled_max_idx = data.wpnts[-1].id
            self.scaled_max_s = data.wpnts[-1].s_m
            self.scaled_delta_s = data.wpnts[1].s_m - data.wpnts[0].s_m

    def updated_wpnts_cb(self, data: WpntArray):
        self.wpnts_updated = data.wpnts[:-1]
        self.max_s_updated = self.wpnts_updated[-1].s_m
        self.max_idx_updated = self.wpnts_updated[-1].id

    def local_wpnts_cb(self, data: WpntArray):
        self.local_wpnts = np.array([[wpnt.s_m, wpnt.d_m] for wpnt in data.wpnts])

    def opponent_trajectory_cb(self, data: OpponentTrajectory):
        self.opponent_waypoints = data.oppwpnts
        self.max_opp_idx = len(data.oppwpnts)-1
        self.opponent_wpnts_sm = np.array([wpnt.s_m for wpnt in data.oppwpnts])

    def ot_sections_check_cb(self, data: Bool):
        self.ot_section_check = data.data

    def dyn_param_cb(self, params: Config):
        # Extract params
        updates = {
            "evasion_dist": rospy.get_param("dynamic_sqp_tuner_node/evasion_dist", 0.65),
            "obs_traj_tresh": rospy.get_param("dynamic_sqp_tuner_node/obs_traj_tresh", 1.5),
            "spline_bound_mindist": rospy.get_param("dynamic_sqp_tuner_node/spline_bound_mindist", 0.2),
            "lookahead": rospy.get_param("dynamic_sqp_tuner_node/lookahead_dist", 15),
            "avoidance_resolution": rospy.get_param("dynamic_sqp_tuner_node/avoidance_resolution", 20),
            "back_to_raceline_before": rospy.get_param("dynamic_sqp_tuner_node/back_to_raceline_before", 5),
            "back_to_raceline_after": rospy.get_param("dynamic_sqp_tuner_node/back_to_raceline_after", 5),
            "avoid_static_obs": rospy.get_param("dynamic_sqp_tuner_node/avoid_static_obs", True)
        }
        
        # Pass updates to core
        self.core.update_params(updates)
        
        print("[Planner] Dynamic reconf triggered. Params updated.")

    def run(self):
        """ Main loop """
        while not rospy.is_shutdown():
            if self.scaled_wpnts is None or self.global_waypoints is None:
                self.rate.sleep()
                continue
            
            # 1. Compute Evasion Trajectory using Core
            # We pass the current state data to the core function
            # Note: You might need to refine 'considered_obs' logic here or inside core
            considered_obs = self.obs.obstacles # Placeholder
            
            evasion_x, evasion_y, evasion_s, evasion_d, evasion_v, no_collision = \
                self.core.compute_evasion_trajectory(
                    considered_obs, self.frenet_state, self.scaled_wpnts_msg, self.scaled_wpnts,
                    self.scaled_max_s, self.scaled_max_idx, self.scaled_delta_s,
                    self.max_s_updated, self.wpnts_updated, self.max_idx_updated,
                    self.opponent_wpnts_sm, self.opponent_waypoints, self.max_opp_idx
                )
            
            # 2. Visualize / Publish Results
            self.visualize_sqp_result(evasion_s, evasion_d, evasion_x, evasion_y, evasion_v)
            
            # 3. Publish OTWpntArray
            evasion_wpnts_msg = OTWpntArray(header=rospy.Header(stamp=rospy.Time.now(), frame_id="map"))
            evasion_wpnts = []
            if len(evasion_x) > 0:
                 evasion_wpnts = [Wpnt(id=i, s_m=s, d_m=d, x_m=x, y_m=y, vx_mps= v) 
                                  for i, (x, y, s, d, v) in enumerate(zip(evasion_x, evasion_y, evasion_s, evasion_d, evasion_v))]
            evasion_wpnts_msg.wpnts = evasion_wpnts
            self.evasion_pub.publish(evasion_wpnts_msg)

            self.rate.sleep()

    def visualize_sqp_result(self, s, d, x, y, v):
        """ Wrapper for visualization logic """
        # Implementation of marker publishing using self.mrks_pub, etc.
        pass

if __name__ == "__main__":
    pass
    # node = DdrxPlannerNode()
    # try:
    #     node.run()
    # except rospy.ROSInterruptException:
    #     pass