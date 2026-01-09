import numpy as np
import casadi as ca
#from typing import Protocol --- LATER ---

#todo: 
# - Test below implementation

from Model import CasadiModel

class CasadiOptimalControlProblem:
    
    def __init__(self,
                model: CasadiModel,
                cost_func: callable,
                params: dict,
                time_horizon: float,
                start_cost: float = 0.0,
                terminal_cost: float = 0.0,
                u_min: float = -10.0,
                u_max: float = 10.0
                ):
        
        self.starting_cost = start_cost
        self.terminal_cost = terminal_cost
        self.time_horizon = time_horizon
        self.cost_func = cost_func
        self.params = params
        self.model = model
        self.u_min = u_min
        self.u_max = u_max

    def solve_single_shooting(self,**kwargs):
        """
        Solves the optimal control problem.

        
        Args:
            **kwargs: Additional keyword arguments for solver configuration.
        """

        N_horizont = int(self.time_horizon / self.model.dt)
        
        n_states = self.model.nStates
        n_controls = self.model.nControls
        
        U = ca.SX.sym('U', n_controls, N_horizont)
        
        # Define symbolic initial state parameter
        X0_sym = ca.SX.sym('X0', n_states)

        cost = 0
        g = []  # Constraints list
        
        # Initialize states with the symbolic initial state
        current_state = X0_sym
        
        for k in range(N_horizont):
            res = self.model.integrator(x0=current_state, p=U[:, k])
            current_state = res['xf']
            
            cost += self.cost_func(self.starting_cost, 
                                    self.terminal_cost, 
                                    current_state, 
                                    U[:, k],
                                    self.params)
            
        nlp = {
            'x': ca.reshape(U, -1, 1), # Decision variables: all control inputs over the horizon, reshaped to a column vector
            'f': cost, # Objective function: total cost to minimize
            'g': ca.SX([]) if not g else ca.vertcat(*g), # Constraints: empty if no real constraints
            'p': X0_sym # Parameters: initial state, passed as parameters to the solver (must be symbolic)
        }

        opts = {
            'ipopt.print_level': 3,  # Debug output
            'print_time': 0,
        }

        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        lbx = [self.u_min] * (n_controls * N_horizont)
        ubx = [self.u_max] * (n_controls * N_horizont)

        # Initialwerte und Referenzwerte
        X0_val = self.model.init_state

        # Löse Optimierungsproblem #todo: make initial guess smarter
        x0_guess  = [0.0] * (n_controls * N_horizont)
        
        result = solver(
            x0=x0_guess,
            lbx=lbx,
            ubx=ubx,
            lbg=[],
            ubg=[],
            p=self.model.init_state
        )

        return np.array(result["x"]).flatten()
    
    def solve_multiple_shooting(self):
        # dt = self.model.dt
        # N = int(self.time_horizon / dt)

        # nx = self.model.nStates
        # nu = self.model.nControls

        # # Entscheidungsvariablen: Zustände an jedem Intervallanfang + Steuerungen
        # X = ca.SX.sym("X", nx, N+1)  # Zustände
        # U = ca.SX.sym("U", nu, N)    # Steuerungen

        # X0_sym = ca.SX.sym("X0", nx)  # Parameter für Anfangszustand

        # cost = 0
        # g = []  # Constraints für Kontinuität

        # for k in range(N):
        #     xk = X[:, k]
        #     uk = U[:, k]

        #     # Integration der Dynamik über ein Intervall
        #     res = self.model
        pass #todo: implement multiple shooting method


if __name__ == "__main__":

    states = ca.SX.sym('x', 2)
    control = ca.SX.sym('u', 1)

    # Define a simple ODE: dx/dt = Ax + Bu
    # System: position and velocity
    A = np.array([[0, 1], [-1, -1]])
    B = np.array([[0], [1]])

    def ode_func(x, u):
        return ca.mtimes(A, x) + ca.mtimes(B, u)

    # Create model with dt=0.1s and total time T=10s
    model = CasadiModel(states=states, controls=control, init_state=[1.0, 0.0], ode=ode_func, dt=0.1, T=10.0)

    # Define a cost function that tracks states to zero with control effort penalty
    def tracking_cost(start, terminal, states, controls, params):
        Q = 10.0  # State tracking weight
        R = 10.0   # Control effort weight
        state_cost = Q * ca.mtimes(states.T, states)  # Minimize state deviation from origin
        control_cost = R * ca.mtimes(controls.T, controls)  # Minimize control effort
        return state_cost + control_cost

    opt_control = CasadiOptimalControlProblem(
        model=model,
        cost_func=tracking_cost,
        params={},
        time_horizon=10.0,
        start_cost=0.0,
        terminal_cost=0.0
    )
    
    result = opt_control.solve_single_shooting()

    import matplotlib.pyplot as plt

    # Extract control inputs from result
    x_solution = np.array(result).flatten()
    N_horizont = int(opt_control.time_horizon / model.dt)
    
    U_opt = x_solution[:model.nControls * N_horizont]
    time_steps = np.linspace(0, opt_control.time_horizon, N_horizont)

    # Simulate the system with optimal control
    x_states = np.zeros((2, N_horizont + 1))
    x_states[:, 0] = model.init_state
    
    for k in range(N_horizont):
        res = model.integrator(x0=x_states[:, k], p=U_opt[k])
        x_states[:, k + 1] = np.array(res['xf']).flatten()

    # Simulate with zero control for comparison
    x_zero = np.zeros((2, N_horizont + 1))
    x_zero[:, 0] = model.init_state
    
    for k in range(N_horizont):
        res = model.integrator(x0=x_zero[:, k], p=0.0)
        x_zero[:, k + 1] = np.array(res['xf']).flatten()

    time_sim = np.linspace(0, opt_control.time_horizon, N_horizont + 1)

    # Create comprehensive plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Position (state 0)
    axes[0, 0].plot(time_sim, x_states[0, :], 'b-o', linewidth=2, markersize=4, label='With Optimal Control')
    axes[0, 0].plot(time_sim, x_zero[0, :], 'r--s', linewidth=2, markersize=4, label='With Zero Control')
    axes[0, 0].axhline(y=0, color='k', linestyle=':', alpha=0.5)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Position x0')
    axes[0, 0].set_title('State 0: Position')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Plot 2: Velocity (state 1)
    axes[0, 1].plot(time_sim, x_states[1, :], 'b-o', linewidth=2, markersize=4, label='With Optimal Control')
    axes[0, 1].plot(time_sim, x_zero[1, :], 'r--s', linewidth=2, markersize=4, label='With Zero Control')
    axes[0, 1].axhline(y=0, color='k', linestyle=':', alpha=0.5)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Velocity x₁')
    axes[0, 1].set_title('State 1: Velocity')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Plot 3: Control Input
    axes[1, 0].bar(time_steps, U_opt, width=0.08, color='green', alpha=0.7, edgecolor='darkgreen', linewidth=1.5)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Control Input u')
    axes[1, 0].set_title('Optimal Control Signal')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Plot 4: State norm (combined tracking error)
    state_norm_opt = np.linalg.norm(x_states, axis=0)
    state_norm_zero = np.linalg.norm(x_zero, axis=0)
    axes[1, 1].plot(time_sim, state_norm_opt, 'b-o', linewidth=2, markersize=4, label='With Optimal Control')
    axes[1, 1].plot(time_sim, state_norm_zero, 'r--s', linewidth=2, markersize=4, label='With Zero Control')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('State Norm ||x||')
    axes[1, 1].set_title('Tracking Error (State Norm)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()

    # Print results
    print("\n=== Optimal Control Results ===")
    print(f"Final state with optimal control: {x_states[:, -1]}")
    print(f"Final state with zero control: {x_zero[:, -1]}")
    print(f"Total control effort: {np.sum(U_opt**2):.4f}")
    print(f"Final state norm (optimal): {state_norm_opt[-1]:.4f}")
    print(f"Final state norm (zero control): {state_norm_zero[-1]:.4f}")
