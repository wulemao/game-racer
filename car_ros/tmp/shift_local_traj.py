import numpy as np
try:
    from mpc_controller import MPCController
except ImportError:
    MPCController = None
    print("Warning: mpc_controller not found. MPC will not work.")

class ShiftLocalTraj:
    def __init__(self, config):
        """
        Initialize with vehicle geometry from config.
        """
        dyn = config['car_dynamics']
        self.LF = dyn['LF']
        self.LR = dyn['LR']
        self.L_base = self.LF + self.LR
        if MPCController is not None:
            self.mpc_controller = MPCController(N=10, dt=0.1, L=self.L_base)
        else:
            self.mpc_controller = None

    # --- MATH & PHYSICS HELPERS ---

    @staticmethod
    def get_curvature(x1, y1, x2, y2, x3, y3):
        """
        Calculate curvature using three points (Menger curvature).
        """
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
        Calculate the raw repulsion factor based on Time-To-Collision (TTC).
        """
        if vs == vs_opp: return 0.
        ttc = (s_opp - s) + (vs_opp - vs) * t
        eff_s = ttc 
        factor = sf1 * np.exp(-sf2 * np.abs(eff_s)**2)
        return factor

    def calculate_avoidance_shift(self, s, e, v, opponents, sf1, sf2, blocking_factor, gap, raceline, raceline_dev, closest_idx, time_horizon=0.5, track_width=0.44):
        """
        Calculate avoidance shift based on opponents. Scalable to N opponents.
        """
        total_shift = 0.0
        
        max_abs_raw_shift = -1.0
        dominant_opp_e = 0.0
        dominant_opp_s = -999.0
        dominant_opp_v = 0.0
        dominant_raw_val = 0.0 # Unsigned raw factor

        N_points = len(raceline)

        # --- Phase 1: Accumulate Shifts & Identify Dominant Threat ---
        for (s_opp, e_opp_raw, v_opp) in opponents:
            # 1. Track wrap-around handling
            diff_s = s_opp - s
            if diff_s < -75.: diff_s += 150.
            if diff_s > 75.: diff_s -= 150.
            
            # 2. Get opponent's e_total (absolute lateral pos)
            idx_offset = int(diff_s / gap)
            opp_raceline_idx = (closest_idx + idx_offset) % N_points
            _e_opp_dev = raceline_dev[opp_raceline_idx]
            e_opp_total = e_opp_raw + _e_opp_dev

            # 3. Calculate Raw Shift (Magnitude)
            raw_factor = self.calc_shift_factor(s, s_opp, v, v_opp, sf1, sf2, t=time_horizon)
            
            # 4. Determine Sign (Direction)
            if e > e_opp_total:
                signed_shift = np.abs(raw_factor)
            else:
                signed_shift = -np.abs(raw_factor)
            
            # 5. Accumulate
            total_shift += signed_shift

            # 6. Track Dominant Opponent
            if np.abs(signed_shift) > max_abs_raw_shift:
                max_abs_raw_shift = np.abs(signed_shift)
                dominant_opp_e = e_opp_total
                dominant_opp_s = s_opp
                dominant_opp_v = v_opp
                dominant_raw_val = raw_factor

        # If no opponents or negligible shift, return early
        if max_abs_raw_shift <= 1e-6:
            return 0.0

        # --- Phase 2: Alignment Logic ---
        # Align target to the dominant opponent's lane
        if (total_shift + dominant_opp_e) * total_shift < 0.:
             total_shift = 0.
        else:
             if max_abs_raw_shift > 0.03:
                 total_shift += dominant_opp_e
        
        # --- Phase 3: Blocking Logic ---
        dist_from_dom = s - dominant_opp_s
        if dist_from_dom < -75.: dist_from_dom += 150.
        if dist_from_dom > 75.:  dist_from_dom -= 150.

        if dist_from_dom > 0: # If dominant opponent is BEHIND me
            bf = 1.0 - np.exp(-blocking_factor * max(dominant_opp_v - v, 0.))
            total_shift = total_shift + (dominant_opp_e - total_shift) * bf * (dominant_raw_val / sf1)

        # --- Phase 4: Boundary Clamping ---
        target_pos = e + total_shift
        
        if target_pos > track_width:
            total_shift = track_width - e
        elif target_pos < -track_width:
            total_shift = -track_width - e
            
        return total_shift

    def get_closest_raceline_idx(self, x, y, raceline, last_i=-1):
        """
        Find the index of the closest point on the raceline.
        """
        if last_i == -1:
            # Global search
            dists = np.sqrt((raceline[:, 0] - x)**2 + (raceline[:, 1] - y)**2)
            return np.argmin(dists)
        else:
            # Local search optimization
            search_len = 20
            indices = np.arange(last_i, last_i + search_len) % len(raceline)
            raceline_window = raceline[indices]
            dists = np.sqrt((raceline_window[:, 0] - x)**2 + (raceline_window[:, 1] - y)**2)
            local_min = np.argmin(dists)
            return indices[local_min]

    # --- CONTROL ALGORITHMS ---
    def mpc(self, xyt, pose, waypoint_generator, *opponents, **kwargs):
        """
        MPC trajectory generation using the internally imported solver.
        
        Args:
            xyt: Tuple (x, y, theta)
            pose: Tuple (s, e, v)
            waypoint_generator: Instance to access raceline data
            *opponents: List of opponent states
            **kwargs: Control parameters
        """
        # [MODIFIED] Use the module-level global variable directly
        if self.mpc_controller is None:
            return 0., 0., 0., 0., kwargs.get('last_i', -1)

        s, e, v = pose
        x, y, theta = xyt

        # Extract parameters
        sf1 = kwargs.get('sf1')
        sf2 = kwargs.get('sf2')
        lookahead_factor = kwargs.get('lookahead_factor')
        v_factor = kwargs.get('v_factor')
        blocking_factor = kwargs.get('blocking_factor')
        gap = kwargs.get('gap', 0.06)
        last_i = kwargs.get('last_i', -1)
        
        raceline = waypoint_generator.raceline
        raceline_dev = waypoint_generator.raceline_dev
        
        closest_idx = self.get_closest_raceline_idx(x, y, raceline, last_i)
        
        _e = raceline_dev[closest_idx]
        e_combined = e + _e
        curr_idx = (closest_idx + 1) % len(raceline)
        next_idx = (curr_idx + 1) % len(raceline)
        traj = []
        dist_target = 0
        
        opp_list = list(opponents)

        for t in np.arange(0.1, 1.05, 0.1): 
            dist_target += v_factor * raceline[curr_idx, 2] * 0.1
            
            # Internal call to avoidance logic
            shift = self.calculate_avoidance_shift(
                s, e_combined, v, opp_list, sf1, sf2, blocking_factor, gap, 
                raceline, raceline_dev, closest_idx, time_horizon=t, track_width=0.44
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
            pt = (1. - ratio) * raceline[next_idx, :2] + ratio * raceline[curr_idx, :2]
            
            theta_traj = np.arctan2(raceline[next_idx, 1] - raceline[curr_idx, 1],
                                    raceline[next_idx, 0] - raceline[curr_idx, 0]) + np.pi / 2.
            
            shifted_pt = pt + shift * np.array([np.cos(theta_traj), np.sin(theta_traj)])
            traj.append(shifted_pt)
            
        try:
            throttle, steer = self.mpc_controller.solve(
                x0=[x, y, theta, v], 
                traj=np.array(traj), 
                lookahead_factor=lookahead_factor
            )
        except Exception:
            throttle, steer = 0.0, 0.0

        lookahead_distance_val = lookahead_factor * raceline[curr_idx, 2]
        lookahead_idx = int(closest_idx + 5) % len(raceline)
        next_idx_h = (lookahead_idx + 1) % len(raceline)

        curv = self.get_curvature(
            raceline[closest_idx-1, 0], raceline[closest_idx-1, 1],
            raceline[closest_idx, 0], raceline[closest_idx, 1],
            raceline[(closest_idx+1)%len(raceline), 0], raceline[(closest_idx+1)%len(raceline), 1]
        )

        curv_lookahead = self.get_curvature(
            raceline[lookahead_idx-1, 0], raceline[lookahead_idx-1, 1],
            raceline[lookahead_idx, 0], raceline[lookahead_idx, 1],
            raceline[next_idx_h, 0], raceline[next_idx_h, 1]
        )
                                                                
        return steer, throttle, curv, curv_lookahead, closest_idx
    
    def pure_pursuit(self, xyt, pose, waypoint_generator, *opponents, **kwargs):
        """
        Pure Pursuit controller.
        Requires 'waypoint_generator' to be passed in.
        """
        s, e, v = pose
        x, y, theta = xyt
        
        sf1 = kwargs.get('sf1')
        sf2 = kwargs.get('sf2')
        lookahead_factor = kwargs.get('lookahead_factor')
        v_factor = kwargs.get('v_factor')
        blocking_factor = kwargs.get('blocking_factor')
        gap = kwargs.get('gap', 0.06)
        last_i = kwargs.get('last_i', -1)

        raceline = waypoint_generator.raceline
        raceline_dev = waypoint_generator.raceline_dev
        
        closest_idx = self.get_closest_raceline_idx(x, y, raceline, last_i)
        
        _e = raceline_dev[closest_idx]
        e_combined = e + _e
        
        shift = self.calculate_avoidance_shift(
            s, e_combined, v, opponents, sf1, sf2, blocking_factor, gap, raceline, raceline_dev, closest_idx, time_horizon=0.5
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
        
        next_idx = (lookahead_idx + 1) % N
        prev_idx = (lookahead_idx - 1) % N
        dx_traj = raceline[next_idx, 0] - raceline[lookahead_idx, 0]
        dy_traj = raceline[next_idx, 1] - raceline[lookahead_idx, 1]
        theta_traj = np.arctan2(dy_traj, dx_traj) + np.pi / 2.
        
        shifted_point = lookahead_point + shift * np.array([np.cos(theta_traj), np.sin(theta_traj), 0.])

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

    # --- COLLISION LOGIC ---

    def calculate_collisions(self, cars, num_cars):
        """
        Calculate collisions and apply decay to car states.
        """
        states = {}
        for i in range(num_cars):
            obs = cars[i]['env'].state
            states[i] = {'px': obs.x, 'py': obs.y, 'theta': obs.psi, 's': cars[i]['s']}

        for i in range(num_cars):
            for j in range(i + 1, num_cars):
                cost = self.has_collided(states[i], states[j])
                if cost > 0:
                    diff_s = states[j]['s'] - states[i]['s']
                    if diff_s < -75.0: diff_s += 150.0
                    elif diff_s > 75.0: diff_s -= 150.0

                    if diff_s > 0: # j ahead
                        self.apply_decay(cars, i, cost, rear=True)
                        self.apply_decay(cars, j, cost, rear=False)
                    else:
                        self.apply_decay(cars, i, cost, rear=False)
                        self.apply_decay(cars, j, cost, rear=True)

    def apply_decay(self, cars, idx, cost, rear):
        """
        Apply velocity decay penalty upon collision.
        """
        f = 20.0 if rear else 5.0
        decay = np.exp(-f * cost)
        cars[idx]['env'].state.vx *= decay
        cars[idx]['env'].state.vy *= decay

    @staticmethod
    def has_collided(s1, s2, L=0.18, B=0.12):
        """
        Check collision between two oriented rectangles.
        """
        dx = s1['px'] - s2['px']
        dy = s1['py'] - s2['py']
        d_long = dx*np.cos(s1['theta']) + dy*np.sin(s1['theta'])
        d_lat = dy*np.cos(s1['theta']) - dx*np.sin(s1['theta'])
        cost = (np.abs(d_long) - 2*L < 0) * (np.abs(d_lat) - 2*B < 0) * 1.0 
        return cost if cost else 0.0
    
    # the interface for ddrx-spliner
    