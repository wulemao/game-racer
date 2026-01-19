import numpy as np
import casadi as ca

class MPCController:
    def __init__(self, N=10, dt=0.1, L=0.34, Sa=0.34, Ta=20./4.65, fr=1.0, Q_weights=[1, 1], R_weights=[1, 1]):
        """
        Initialize the MPC Controller, building the CasADi solver instance.
        
        Args:
            N (int): Prediction horizon steps.
            dt (float): Time step in seconds.
            L (float): Wheelbase of the vehicle.
            Sa (float): Steering actuator coefficient.
            Ta (float): Throttle actuator coefficient.
            fr (float): Friction/Resistance coefficient.
            Q_weights (list): Weights for state error [lateral_error, heading_error].
            R_weights (list): Weights for control input [acceleration, steering].
        """
        self.N = N
        self.dt = dt
        self.L = L
        
        # --- 1. Define Symbolic Variables ---
        # State variables: [x, y, psi, v] size: 4 x (N+1)
        x = ca.MX.sym('x', 4, N+1) 
        # Control inputs: [acceleration, steering_angle] size: 2 x N
        u = ca.MX.sym('u', 2, N)
        # Initial state parameter
        x0_ = ca.MX.sym('x0', 4)
        
        # Target state parameter: flattened array [x_0...x_N, y_0...y_N]
        target_state = ca.MX.sym('target_states', N*2) 
        
        # Dynamic weight parameter (used to adjust cost based on lookahead)
        qr_ratio = ca.MX.sym('qr_ratio', 1)

        # Weight matrices
        # Penalize lateral error (e) and heading error (psi)
        Q = np.diag(Q_weights) / qr_ratio[0]
        # Penalize control inputs (a, delta)
        R = np.diag(R_weights)

        # --- 2. Build Cost Function and Constraints ---
        cost = 0
        g = [] # Constraints list
        
        # Initial state constraint (Equality)
        g.append(x[:, 0] - x0_)

        for k in range(N):
            # Dynamics constraints (Bicycle Model)
            x_next = self.vehicle_dynamics(x[:, k], u[:, k], L, Sa, Ta, fr, dt)
            g.append(x[:, k+1] - x_next)
            
            # Calculate cost function: state deviation
            # Note: target_state structure is [all_x, all_y]
            ref_x = target_state[k]
            ref_y = target_state[N+k]
            
            # Error vector: [x - ref_x, y - ref_y]
            # Note: In the original logic, this was used as a proxy for lateral/heading error
            state_error = ca.vertcat(x[0, k] - ref_x, x[1, k] - ref_y)
            
            cost += ca.mtimes([state_error.T, Q, state_error])  # State cost
            cost += ca.mtimes([u[:, k].T, R, u[:, k]])          # Control cost

        # Additional Physical Constraints
        
        # Lateral acceleration constraint: v^2 * tan(delta) / L
        # This approximates lateral acceleration limits to prevent slipping
        for k in range(N):
            g.append((x[3, k]**2) * ca.tan(u[1, k]))
            
        # Velocity constraints (e.g., max speed)
        for k in range(N):
            g.append(x[3, k+1])

        # --- 3. Create NLP Solver ---
        # Flatten variables for the solver
        opt_variables = ca.vertcat(ca.reshape(x, -1, 1), ca.reshape(u, -1, 1))
        opt_constraints = ca.vertcat(*g)
        
        # Define NLP problem
        nlp_prob = {
            'f': cost, 
            'x': opt_variables, 
            'g': opt_constraints, 
            'p': ca.vertcat(x0_, target_state, qr_ratio)
        }
        
        # Solver options (Ipopt)
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.max_iter': 500}
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

        # --- 4. Initialize Constraint Bounds ---
        # Total constraints = Initial(4) + Dynamics(4*N) + LatAcc(N) + Vel(N)
        self.n_constraints = 4 * (N + 1) + 2 * N 
        self.lbg = np.zeros((self.n_constraints, 1))
        self.ubg = np.zeros((self.n_constraints, 1))

    @staticmethod
    def vehicle_dynamics(xk, uk, L, Sa, Ta, fr, dt):
        """
        Static helper for kinematic bicycle model dynamics.
        """
        x, y, psi, v = xk[0], xk[1], xk[2], xk[3]
        a, delta = uk[0], uk[1]
        
        x_next = ca.vertcat(
            x + v * ca.cos(psi) * dt,
            y + v * ca.sin(psi) * dt,
            psi + (v / L) * ca.tan(Sa * delta) * dt,
            v + (Ta * a - fr) * dt
        )
        return x_next

    def solve(self, x0, traj, lookahead_factor=1.0, mu=1.6, g=9.81):
        """
        Execute the MPC solver for the current state.

        Args:
            x0: Current state [x, y, psi, v].
            traj: Reference trajectory points.
            lookahead_factor: Factor to adjust weight matrices dynamically.
            mu: Friction coefficient.
            g: Gravity constant.

        Returns:
            Tuple (acceleration, steering_angle)
        """
        # Ensure trajectory is flattened correctly [x1...xN, y1...yN]
        if traj.shape == (self.N, 2):
            traj_flat = traj.T.flatten()
        else:
            traj_flat = traj.flatten() 

        # Prepare parameter vector: [x0, target_state, qr_ratio]
        params = np.concatenate((x0, traj_flat, np.array([lookahead_factor])))

        # Initial guess (warm start could be implemented here)
        x_init = np.tile(x0, (self.N + 1, 1)).T
        u_init = np.zeros((2, self.N))
        initial_guess = ca.vertcat(ca.reshape(x_init, -1, 1), ca.reshape(u_init, -1, 1))

        # Update Constraint Bounds
        # 1. Lateral Acceleration Constraints (Indices: -2*N to -N)
        # Limit ~ mu * g * L
        idx_lat_start = -2 * self.N
        idx_lat_end = -self.N
        lat_limit = mu * g * self.L
        
        self.ubg[idx_lat_start:idx_lat_end] = lat_limit
        self.lbg[idx_lat_start:idx_lat_end] = -lat_limit
        
        # 2. Velocity Constraints (Indices: -N to end)
        # Max speed limit (e.g., 5.0 m/s as in original code)
        idx_vel_start = -self.N
        self.ubg[idx_vel_start:] = 5.0   
        self.lbg[idx_vel_start:] = -100.0 # Loose lower bound

        # Solve NLP
        solution = self.solver(x0=initial_guess, lbg=self.lbg, ubg=self.ubg, p=params)

        # Extract optimal control inputs
        optimal_solution = solution['x']
        # State variables are first 4*(N+1), Control inputs follow
        idx_u_start = 4 * (self.N + 1)
        optimal_u = ca.reshape(optimal_solution[idx_u_start:], 2, self.N).full()
        
        # Return the first control action [acc, steer]
        u_curr = optimal_u[:, 0]
        return float(u_curr[0]), float(u_curr[1])