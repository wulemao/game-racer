import numpy as np
import osqp
import scipy.sparse as sp
import matplotlib.pyplot as plt
import time

class SoftSplineFitN:
    def __init__(self, n_segments=3):
        self.n_segments = n_segments
        
        # Data containers
        self.x_data = None
        self.y_data = None
        self.t_data = None
        
        # Hard Constraints (Start and End states)
        self.start_state = {} 
        self.end_state = {}   
        
        # Breakpoints
        self.breakpoints = None
        
        # Optimized coefficients
        self.coeffs_opt = None 

    def set_data(self, x_data, y_data, t_data, start_state, end_state, breakpoints=None):
        self.x_data = np.array(x_data)
        self.y_data = np.array(y_data)
        self.t_data = np.array(t_data)
        self.start_state = start_state
        self.end_state = end_state
        
        t0 = start_state['t']
        tf = end_state['t']
        
        if breakpoints is None:
            self.breakpoints = np.linspace(t0, tf, self.n_segments + 1)
        else:
            self.breakpoints = np.array(breakpoints)
            self.n_segments = len(self.breakpoints) - 1

    # --- Helper: Design Matrices ---
    def _basis(self, t, order=0):
        # order 0: Position, 1: Velocity, 2: Acceleration
        if order == 0:   # Pos: [t^5, t^4, t^3, t^2, t, 1]
            return np.array([t**5, t**4, t**3, t**2, t, 1.0])
        elif order == 1: # Vel: [5t^4, 4t^3, 3t^2, 2t, 1, 0]
            return np.array([5*t**4, 4*t**3, 3*t**2, 2*t, 1.0, 0.0])
        elif order == 2: # Acc: [20t^3, 12t^2, 6t, 2, 0, 0]
            return np.array([20*t**3, 12*t**2, 6*t, 2.0, 0.0, 0.0])
        return np.zeros(6)

    def _get_segment_idx(self, t):
        if t >= self.breakpoints[-1]:
            return self.n_segments - 1
        if t <= self.breakpoints[0]:
            return 0
        idx = np.searchsorted(self.breakpoints, t, side='right') - 1
        return max(0, min(idx, self.n_segments - 1))

    def build_qp_matrices(self, fit_weight=1.0, reg_weight=1e-6):
        n_vars_dim = 6 * self.n_segments
        n_vars_total = 2 * n_vars_dim
        
        Q = np.zeros((n_vars_total, n_vars_total))
        c = np.zeros(n_vars_total)
        
        # 1. Soft Constraints (Least Squares)
        for i, t in enumerate(self.t_data):
            seg_idx = self._get_segment_idx(t)
            T = self._basis(t, order=0)
            
            # X Dimension
            idx_x = seg_idx * 6 
            Q[idx_x:idx_x+6, idx_x:idx_x+6] += 2 * fit_weight * np.outer(T, T)
            c[idx_x:idx_x+6] += -2 * fit_weight * self.x_data[i] * T
            
            # Y Dimension
            idx_y = n_vars_dim + (seg_idx * 6)
            Q[idx_y:idx_y+6, idx_y:idx_y+6] += 2 * fit_weight * np.outer(T, T)
            c[idx_y:idx_y+6] += -2 * fit_weight * self.y_data[i] * T

        Q += np.eye(n_vars_total) * reg_weight

        # 2. Hard Constraints
        A_rows = []
        b_vals = []
        
        def add_constr(coeffs_row, value):
            A_rows.append(coeffs_row)
            b_vals.append(value)

        t0 = self.breakpoints[0]
        tf = self.breakpoints[-1]
        
        for dim in range(2):
            dim_offset = 0 if dim == 0 else n_vars_dim
            
            s_key = 'x' if dim == 0 else 'y'
            v_key = 'vx' if dim == 0 else 'vy'
            # a_key no longer needed for boundary
            
            val_start_p = self.start_state.get(s_key, 0.0)
            val_start_v = self.start_state.get(v_key, 0.0)
            
            val_end_p = self.end_state.get(s_key, 0.0)
            val_end_v = self.end_state.get(v_key, 0.0)

            # --- A. Start State (Only Pos and Vel) ---
            # Position
            row = np.zeros(n_vars_total)
            row[dim_offset : dim_offset+6] = self._basis(t0, 0)
            add_constr(row, val_start_p)
            
            # Velocity
            row = np.zeros(n_vars_total)
            row[dim_offset : dim_offset+6] = self._basis(t0, 1)
            add_constr(row, val_start_v)
            
            # [REMOVED] Start Acceleration Constraint
            # We allow acceleration to be whatever is needed to fit the data

            # --- B. End State (Only Pos and Vel) ---
            last_seg_offset = dim_offset + (self.n_segments - 1) * 6
            # Position
            row = np.zeros(n_vars_total)
            row[last_seg_offset : last_seg_offset+6] = self._basis(tf, 0)
            add_constr(row, val_end_p)
            
            # Velocity
            row = np.zeros(n_vars_total)
            row[last_seg_offset : last_seg_offset+6] = self._basis(tf, 1)
            add_constr(row, val_end_v)
            
            # [REMOVED] End Acceleration Constraint

            # --- C. Continuity Constraints (Internal breakpoints) ---
            # Keep Position, Velocity, AND Acceleration continuity
            for k in range(self.n_segments - 1):
                t_switch = self.breakpoints[k+1]
                curr_offset = dim_offset + k * 6
                next_offset = dim_offset + (k + 1) * 6
                
                # Enforce Continuity for Pos(0), Vel(1), Acc(2)
                for order in range(3):
                    row = np.zeros(n_vars_total)
                    basis_vec = self._basis(t_switch, order)
                    row[curr_offset : curr_offset+6] = basis_vec
                    row[next_offset : next_offset+6] = -basis_vec
                    add_constr(row, 0.0)

        return Q, c, np.array(A_rows), np.array(b_vals)

    def solve(self, fit_weight=100.0):
        # Build matrices
        Q, c, A_eq, b_eq = self.build_qp_matrices(fit_weight=fit_weight)
        
        Q_s = sp.csc_matrix(Q)
        A_s = sp.csc_matrix(A_eq)
        
        prob = osqp.OSQP()
        prob.setup(P=Q_s, q=c, A=A_s, l=b_eq, u=b_eq, verbose=False)
        
        res = prob.solve()
        
        if res.info.status != 'solved':
            print(f"QP Solver Failed: {res.info.status}")
            return False
            
        self.coeffs_opt = res.x
        return True

    def get_trajectory(self, dt=0.05):
        if self.coeffs_opt is None: return None, None, None
        
        t_eval = np.arange(self.breakpoints[0], self.breakpoints[-1] + dt/10.0, dt)
        x_eval, y_eval = [], []
        n_dim_vars = 6 * self.n_segments
        
        for t in t_eval:
            seg_idx = self._get_segment_idx(t)
            T = np.array([t**5, t**4, t**3, t**2, t, 1])
            
            idx_x = seg_idx * 6
            c_x = self.coeffs_opt[idx_x : idx_x+6]
            
            idx_y = n_dim_vars + (seg_idx * 6)
            c_y = self.coeffs_opt[idx_y : idx_y+6]
            
            x_eval.append(np.dot(c_x, T))
            y_eval.append(np.dot(c_y, T))
            
        return np.array(x_eval), np.array(y_eval), t_eval

    def visualize(self, filename="result.png"):
        if self.coeffs_opt is None: return

        x, y, t = self.get_trajectory(dt=0.02)
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.x_data, self.y_data, 'rx', markersize=10, label='Control Points')
        plt.plot(x, y, 'b-', linewidth=2, label=f'Soft Spline (N={self.n_segments})')
        plt.plot(self.start_state['x'], self.start_state['y'], 'go', label='Start')
        plt.plot(self.end_state['x'], self.end_state['y'], 'ro', label='End')
        
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.title(f"Spline Fit (No Accel Boundary Constraints) - N={self.n_segments}")
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.savefig(filename, dpi=300)
        print(f"Saved visualization to {filename}")
        plt.close()

if __name__ == "__main__":
    # 1. Zig-zag data
    ctrl_x = [0, 10, 20, 30, 40]
    ctrl_y = [0,  2,  5,  2,  0]
    ctrl_t = [0,  1,  2,  3,  4]
    
    # 2. Hard Constraints (Only Pos and Vel will be enforced now)
    start = {'x': 0,  'y': 0, 'vx': 10, 'vy': 0, 't': 0}
    end   = {'x': 40, 'y': 0, 'vx': 10, 'vy': 0, 't': 4}
    
    # 3. Test with N=1 to prove it can now curve!
    print("--- Testing with N=1 Segment ---")
    # --- Timing the Solve Process ---
    t_start = time.perf_counter()
    fitter = SoftSplineFitN(n_segments=2)
    fitter.set_data(ctrl_x, ctrl_y, ctrl_t, start, end)
    
    # Use a decent weight to encourage fitting
    success = fitter.solve(fit_weight=100.0)
    t_end = time.perf_counter()
        
    solve_ms = (t_end - t_start) * 1000.0 
    
    if success:
        print(f"Solver Success! Time: {solve_ms:.3f} ms")
        fitter.visualize("fit_n1_released.png")
    else:
        print("Solver failed.")