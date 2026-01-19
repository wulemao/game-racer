import numpy as np
import matplotlib.pyplot as plt
import os
import time

# Assuming SoftSplineFitN is available here
from spline_fit import SoftSplineFitN 

class PolynomialTraj:
    def __init__(self, coeffs_x, coeffs_y, breakpoints, control_points, 
                 L, dt, n_segments, max_s_updated, converter, initial_ref_spline):
        """
        Pure trajectory calculation class.
        It does not contain the optimization solver; it is responsible only for 
        calculating states based on the given coefficients.
        
        :param coeffs_x: (N_segments, 6) shape numpy array, polynomial coefficients for X-axis
        :param coeffs_y: (N_segments, 6) shape numpy array, polynomial coefficients for Y-axis
        :param breakpoints: List of time breakpoints [t0, t1, ... tN]
        :param control_points: Original control points matrix [x, y, t] (for visualization or reference)
        :param L: Wheelbase
        :param dt: Sampling time step
        :param n_segments: Number of segments
        :param max_s_updated: Max S value of the track (for loop handling)
        :param converter: Coordinate converter instance
        :param initial_ref_spline: Reference line instance
        """
        # 1. Receive optimization results
        self.coeffs_x = coeffs_x
        self.coeffs_y = coeffs_y
        self.breakpoints = breakpoints
        self.P = control_points
        self.t_total = self.P[-1, 2] # End time

        # 2. Receive configuration parameters
        self.L = L
        self.dt = dt
        self.n_segments = n_segments
        self.max_s = max_s_updated
        self.converter = converter
        self.initial_ref_spline = initial_ref_spline

    # --- Core Engine: Vectorized Polynomial Calculation ---

    def _calc_poly_val(self, coeffs, t):
        """ Calculate polynomial value (Position) """
        return (coeffs[0]*t**5 + coeffs[1]*t**4 + coeffs[2]*t**3 + 
                coeffs[3]*t**2 + coeffs[4]*t + coeffs[5])

    def _calc_poly_d1(self, coeffs, t):
        """ Calculate 1st derivative (Velocity) """
        return (5*coeffs[0]*t**4 + 4*coeffs[1]*t**3 + 3*coeffs[2]*t**2 + 
                2*coeffs[3]*t + coeffs[4])

    def _calc_poly_d2(self, coeffs, t):
        """ Calculate 2nd derivative (Acceleration) """
        return (20*coeffs[0]*t**3 + 12*coeffs[1]*t**2 + 6*coeffs[2]*t + 2*coeffs[3])

    def _get_batch_val(self, t_batch, order=0):
        """
        Automatically index the corresponding segment based on time batch and calculate values.
        """
        t_batch = np.array(t_batch)
        val_x = np.zeros_like(t_batch)
        val_y = np.zeros_like(t_batch)
        
        # Iterate through each segment for Mask calculation
        for k in range(self.n_segments):
            t_start = self.breakpoints[k]
            t_end = self.breakpoints[k+1]
            
            # Mask logic: Include right boundary for the last segment; 
            # for others, include left but exclude right [start, end)
            if k == self.n_segments - 1:
                mask = (t_batch >= t_start) & (t_batch <= t_end + 1e-6)
            else:
                mask = (t_batch >= t_start) & (t_batch < t_end)
            
            if not np.any(mask):
                continue
                
            cx = self.coeffs_x[k]
            cy = self.coeffs_y[k]
            t_valid = t_batch[mask]
            
            if order == 0:
                val_x[mask] = self._calc_poly_val(cx, t_valid)
                val_y[mask] = self._calc_poly_val(cy, t_valid)
            elif order == 1:
                val_x[mask] = self._calc_poly_d1(cx, t_valid)
                val_y[mask] = self._calc_poly_d1(cy, t_valid)
            elif order == 2:
                val_x[mask] = self._calc_poly_d2(cx, t_valid)
                val_y[mask] = self._calc_poly_d2(cy, t_valid)
                
        return val_x, val_y

    # --- Differential Flatness State Derivation ---

    def calculate_at_t_batch(self, t_batch):
        """
        Compute full state [x, y, v, phi] and input [a, delta]
        """
        bx, by = self._get_batch_val(t_batch, order=0)
        dbx, dby = self._get_batch_val(t_batch, order=1)
        ddbx, ddby = self._get_batch_val(t_batch, order=2)
        
        # Derived quantities
        v = np.hypot(dbx, dby)
        phi = np.arctan2(dby, dbx)
        
        # Tangential acceleration a_tan = (v')
        a_tan = (dbx * ddbx + dby * ddby) / (v + 1e-6)
        
        # Curvature Kappa
        num = dbx * ddby - dby * ddbx
        den = v**3 + 1e-6
        kappa = num / den
        
        # Front wheel steering angle
        delta = np.arctan(self.L * kappa)
        
        states_batch = np.column_stack((bx, by, v, phi))
        inputs_batch = np.column_stack((a_tan, delta))
        
        return states_batch, inputs_batch

    # --- Public Interface: Get Frenet Trajectory ---

    def get_df_frenet(self):
        """
        Resample the entire trajectory, calculate differential flatness states, 
        and convert to Frenet frame.
        Note: Does not accept evasion_s/d arguments anymore; uses trajectory parameters
        passed during initialization.
        """
        # 1. Determine sampling points
        # Determine number of points based on total time and dt, adding a buffer at ends
        interp_nums = int(self.t_total / self.dt) + 5
        t_samples = np.linspace(0, self.t_total, interp_nums)
        
        # 2. Calculate Cartesian states (Batch)
        start_flat = time.time()
        states_list, inputs_list = self.calculate_at_t_batch(t_samples)
        end_flat = time.time()
        
        # 3. Convert to Frenet frame
        start_conv = time.time()
        
        # Assume converter supports vectorized input
        resp = self.converter.get_frenet(states_list[:, 0], states_list[:, 1])
        s_traj, d_traj = resp[0, :].flatten(), resp[1, :].flatten()
        
        # Calculate Delta Phi (Heading error)
        # Optimized if initial_ref_spline supports vectorization, otherwise keep loop
        delta_phi = np.zeros(len(s_traj))
        for i, (si, phi) in enumerate(zip(s_traj, states_list[:, 3])):
            delta_phi[i] = self.initial_ref_spline.get_delta_phi(si, phi)
            
        end_conv = time.time()
        
        # 4. Assemble final states
        frenet_states = np.column_stack((s_traj, d_traj, states_list[:, 2], delta_phi))
        
        print(f"[PolyTraj] Flatness Calc: {(end_flat - start_flat)*1000:.3f} ms")
        print(f"[PolyTraj] Frenet Conv:   {(end_conv - start_conv)*1000:.3f} ms")
        
        return frenet_states, inputs_list

    def visualize(self, filename="poly_traj.png"):
        t_fine = np.linspace(0, self.t_total, 200)
        states, _ = self.calculate_at_t_batch(t_fine)
        x_fit, y_fit = states[:, 0], states[:, 1]
        
        plt.figure(figsize=(10, 8))
        plt.plot(self.P[:, 0], self.P[:, 1], 'rx', markersize=8, label='Control Points')
        plt.plot(x_fit, y_fit, 'b-', linewidth=2, label=f'Trajectory (N={self.n_segments})')
        plt.title("Polynomial Trajectory")
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        
        save_dir = os.path.dirname(filename)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(filename)
        plt.close()


# ==============================================================================
# For Usage: PolynomialTraj Instance
# ==============================================================================

def plan_polynomial_trajectory(
    evasion_x, evasion_y, evasion_t, 
    start_vx, start_vy, end_vx, end_vy,
    L, dt, n_segments, max_s_updated, converter, initial_ref_spline
):
    """
    Executes QP optimization. If successful, returns an initialized PolynomialTraj object.
    
    :return: PolynomialTraj instance if success, else None
    """
    # 1. Construct control point matrix P [x, y, t]
    if len(evasion_x) != len(evasion_y) or len(evasion_x) != len(evasion_t):
        print("Error: Input arrays length mismatch.")
        return None
        
    P = np.column_stack((evasion_x, evasion_y, evasion_t))
    
    # 2. Construct boundary conditions
    start_state = {
        'x': P[0, 0], 'y': P[0, 1], 't': P[0, 2],
        'vx': start_vx, 'vy': start_vy
    }
    end_state = {
        'x': P[-1, 0], 'y': P[-1, 1], 't': P[-1, 2],
        'vx': end_vx, 'vy': end_vy
    }
    
    # 3. Instantiate and run QP solver
    # Note: SoftSplineFitN logic is in another file, only called here
    fitter = SoftSplineFitN(n_segments=n_segments)
    fitter.set_data(
        x_data=P[:, 0], 
        y_data=P[:, 1], 
        t_data=P[:, 2], 
        start_state=start_state, 
        end_state=end_state
    )
    
    t_start = time.perf_counter()
    success = fitter.solve(fit_weight=1000.0) # Weight is adjustable
    t_end = time.perf_counter()
    solve_ms = (t_end - t_start) * 1000.0 
    print(f"Solver Success! Time: {solve_ms:.3f} ms")

    if not success:
        print("[Planner] QP Solver Failed!")
        return None
        
    # 4. Parse coefficients (Reshape)
    # Convert 1D result to (N, 6) for use in PolynomialTraj
    raw_coeffs = fitter.coeffs_opt
    n_vars_dim = 6 * n_segments
    
    coeffs_x = raw_coeffs[0 : n_vars_dim].reshape(n_segments, 6)
    coeffs_y = raw_coeffs[n_vars_dim : 2 * n_vars_dim].reshape(n_segments, 6)
    
    breakpoints = fitter.breakpoints
    
    # print(f"[Planner] Solved in {(end_solve-start_solve)*1000:.2f} ms")
    
    # 5. Instantiate PolynomialTraj and return
    traj_instance = PolynomialTraj(
        coeffs_x=coeffs_x,
        coeffs_y=coeffs_y,
        breakpoints=breakpoints,
        control_points=P,
        L=L, dt=dt, n_segments=n_segments,
        max_s_updated=max_s_updated,
        converter=converter,
        initial_ref_spline=initial_ref_spline
    )
    
    return traj_instance

# ==============================================================================
# Usage Example
# ==============================================================================
if __name__ == "__main__":
    # Mock data
    raw_x = [0, 10, 20, 30, 40]
    raw_y = [0,  2,  5,  2,  0]
    raw_t = [0,  1,  2,  3,  4]
    
    # Mock external objects
    class MockConverter:
        def get_frenet(self, x, y): 
            # Mock conversion, simply return x=s, y=d
            return np.array([x, y])
            
    class MockRefSpline:
        def get_delta_phi(self, s, phi): return 0.0
        
    mock_conv = MockConverter()
    mock_ref = MockRefSpline()
    
    # 1. Call external function for planning
    traj = plan_polynomial_trajectory(
        evasion_x=raw_x, evasion_y=raw_y, evasion_t=raw_t,
        start_vx=10, start_vy=0, end_vx=10, end_vy=0,
        L=2.8, dt=0.05, n_segments=2, max_s_updated=100,
        converter=mock_conv, initial_ref_spline=mock_ref
    )
    
    if traj:
        # 2. If successful, use the traj object to get Frenet states directly
        frenet_states, inputs = traj.get_df_frenet()
        
        print("Frenet States Shape:", frenet_states.shape)
        traj.visualize("refactored_traj.png")