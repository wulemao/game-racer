import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml
import time
import os

_THIS_DIR = os.path.dirname(os.path.realpath(__file__))
_GAME_RACER_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))

def get_curvature(x1, y1, x2, y2, x3, y3):
    """Calculates signed curvature from three points using NumPy/JAX compatible logic."""
    # Ensure inputs are treated as arrays for vectorized operations if needed
    a = jnp.sqrt((x2-x1)**2 + (y2-y1)**2)
    b = jnp.sqrt((x3-x2)**2 + (y3-y2)**2)
    c = jnp.sqrt((x3-x1)**2 + (y3-y1)**2)
    s = (a + b + c) / 2.0
    area = jnp.sqrt(jnp.maximum(0, s * (s - a) * (s - b) * (s - c)))
    # Determinant for sign
    prod = (x2 - x1) * (y3 - y2) - (x3 - x2) * (y2 - y1)
    return (4.0 * area / (a * b * c + 1e-8)) * jnp.sign(prod)

def custom_fn(theta, traj):
    """Linearly interpolates position, velocity, and curvature from trajectory data."""
    max_theta = traj[-1, 0]
    theta = jnp.mod(theta, max_theta)
    
    idx = jnp.searchsorted(traj[:, 0], theta, side='right') - 1
    idx = jnp.clip(idx, 0, len(traj) - 2)
    
    ratio = (theta - traj[idx, 0]) / (traj[idx + 1, 0] - traj[idx, 0])
    
    pos = traj[idx, 1:3] + ratio * (traj[idx + 1, 1:3] - traj[idx, 1:3])
    v = traj[idx, 3] + ratio * (traj[idx + 1, 3] - traj[idx, 3])
    
    N = len(traj)
    # Using specific index calculation instead of passing large blocks
    curv = get_curvature(
        traj[idx, 1], traj[idx, 2],
        traj[(idx + 1) % N, 1], traj[(idx + 1) % N, 2],
        traj[(idx + 2) % N, 1], traj[(idx + 2) % N, 2]
    )
    return pos, v, {'curv': curv}

class WaypointGenerator:
    def __init__(self, waypoint_type: str, dt: float, H: int, speed: float):
        yaml_path = waypoint_type if waypoint_type.endswith('.yaml') else None
        if not yaml_path:
            raise ValueError("Refactored Generator expects a .yaml track config.")
            
        with open(yaml_path, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
            
        track_cfg = cfg['track_info']
        centerline = track_cfg['centerline_file'][:-4]
        ox, oy, scale = track_cfg['ox'], track_cfg['oy'], float(track_cfg['scale'])
        
        rl_path = os.path.join(_GAME_RACER_ROOT, "ref_trajs", f"{centerline}_raceline_with_speeds.csv")
        df_rl = pd.read_csv(rl_path)
        self.raceline = (df_rl.iloc[:, 1:4].values + np.array([ox, oy, 0]))
        self.raceline[:, :2] *= scale
        self.raceline_dev = 2.0 * df_rl.iloc[:, 4].values
        
        ref_name = f"{centerline}_with_speeds.csv"
        ref_path = os.path.join(_GAME_RACER_ROOT, "ref_trajs", ref_name)
        df_ref = pd.read_csv(ref_path)
        
        if 'num' in waypoint_type:
            self.path = df_ref.iloc[:-1, :].values * scale + np.array([0, ox, oy, 0])
        else:
            self.path = df_ref.values + np.array([0, ox, oy, 0])
        
        # Convert path to JAX array for consistent indexing in generate()
        self.path = jnp.array(self.path)
        self.path = self.path.at[:, -1].set(speed) 
        
        self.waypoint_type = 'custom'
        self.fn = custom_fn
        
        width = track_cfg['track_width']
        # Boundary calculations remain in NumPy during init is fine
        self.left_boundary = self.calc_shifted_traj(np.array(self.path[:, 1:]), width/2. + 0.15)
        self.right_boundary = self.calc_shifted_traj(np.array(self.path[:, 1:]), -width/2. - 0.2)
        
        self.scale_val = 6.5 * (float(cfg['vehicle_params']['Lf']) + float(cfg['vehicle_params']['Lr']))
        self.waypoint_t_list = jnp.arange(0, self.path[-1, 0], 0.1 * self.scale_val)
        
        # Vectorized pre-computation
        self.waypoint_list = jnp.array([self.fn(t, self.path)[0] for t in self.waypoint_t_list])
        self.waypoint_list_np = np.array(self.waypoint_list)
        
        self.dt, self.H, self.speed = dt, H, speed
        self.last_i = -1

    def calc_shifted_traj(self, traj, shift_dist):
        next_pts = np.roll(traj, -1, axis=0)
        prev_pts = np.roll(traj, 1, axis=0)
        yaws = np.arctan2(next_pts[:, 1] - prev_pts[:, 1], next_pts[:, 0] - prev_pts[:, 0])
        shifted = np.copy(traj)
        shifted[:, 0] += shift_dist * np.cos(yaws + np.pi/2)
        shifted[:, 1] += shift_dist * np.sin(yaws + np.pi/2)
        return shifted

    def line_point_distance(self, px, py, x1, y1, x2, y2):
        dx, dy = x2 - x1, y2 - y1
        # Use jnp.where to avoid potential division by zero and maintain JAX compatibility
        denom = dx*dx + dy*dy
        return jnp.where(denom == 0, 0.0, ((px - x1) * dx + (py - y1) * dy) / denom)

    def generate(self, obs: jnp.ndarray, dt=-1, mu_factor=1., body_speed=1.) -> jnp.ndarray:
        dt = self.dt if dt < 0 else dt
        pos2d = obs[:2]
        
        # 1. Sliding Window Nearest Neighbor Search
        if self.last_i == -1:
            dists = jnp.linalg.norm(self.waypoint_list - pos2d, axis=-1)
            t_idx = jnp.argmin(dists)
        else:
            search_end = min(self.last_i + 20, len(self.waypoint_list))
            local_pts = self.waypoint_list[self.last_i:search_end]
            dists = jnp.linalg.norm(local_pts - pos2d, axis=-1)
            t_idx = jnp.argmin(dists) + self.last_i
            
            if self.last_i + 20 >= len(self.waypoint_list):
                wrap_pts = self.waypoint_list[:20]
                wrap_dists = jnp.linalg.norm(wrap_pts - pos2d, axis=-1)
                if jnp.min(wrap_dists) < jnp.min(dists):
                    t_idx = jnp.argmin(wrap_dists)

        self.last_i = t_idx
        t_closed = self.waypoint_t_list[t_idx]
        
        # 2. Refine projection (Distance to segments)
        N = len(self.waypoint_list)
        
        # --- FIX: Changed from [...] to jnp.array([...]) to avoid TypeError ---
        idx_array = jnp.array([(t_idx-1)%N, t_idx, (t_idx+1)%N])
        pts = self.waypoint_list[idx_array]
        
        def get_dist_and_shift(p1, p2):
            t = self.line_point_distance(pos2d[0], pos2d[1], p1[0], p1[1], p2[0], p2[1])
            proj_pt = p1 + jnp.clip(t, 0, 1) * (p2 - p1)
            dist = jnp.linalg.norm(pos2d - proj_pt)
            side = jnp.sign((p2[0]-p1[0])*(pos2d[1]-p1[1]) - (p2[1]-p1[1])*(pos2d[0]-p1[0]))
            return dist * side, t

        dist1, t1 = get_dist_and_shift(pts[0], pts[1])
        dist2, t2 = get_dist_and_shift(pts[1], pts[2])
        
        # Use jnp.abs and jnp.where for cleaner logic
        if jnp.abs(dist1) < jnp.abs(dist2):
            final_dist = dist1
            seg_len = jnp.linalg.norm(pts[1] - pts[0])
            t_closed_refined = t_closed + seg_len * (jnp.clip(t1, 0, 1) - 1.0)
        else:
            final_dist = dist2
            seg_len = jnp.linalg.norm(pts[2] - pts[1])
            t_closed_refined = t_closed + seg_len * jnp.clip(t2, 0, 1)

        # 3. Horizon Prediction
        speed_sqrt_mu = jnp.sqrt(mu_factor)
        _, current_speed, _ = self.fn(t_closed_refined, self.path)
        current_speed *= speed_sqrt_mu
        
        kin_pos, _, _ = self.fn(t_closed_refined + 1.2 * current_speed, self.path)
        
        indices = jnp.arange(self.H + 1)
        t_horizons = t_closed_refined + indices * dt * body_speed
        
        def get_state(t_val):
            p, v, info = self.fn(t_val, self.path)
            p_next, _, _ = self.fn(t_val + dt * v * speed_sqrt_mu, self.path)
            psi = jnp.arctan2(p_next[1] - p[1], p_next[0] - p[0])
            # Explicitly extract the curvature value from the info dict
            curv_val = info['curv']
            return jnp.array([p[0], p[1], psi, curv_val, v * speed_sqrt_mu])

        target_pos_list = jax.vmap(get_state)(t_horizons)
        
        return target_pos_list, kin_pos, t_closed_refined, final_dist