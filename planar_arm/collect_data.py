import numpy as np
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy import sparse
import random
import os, torch
import copy
import osqp
import matplotlib.animation as animation
# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define the hyperrectangle class (similar to LazySets.Hyperrectangle in Julia)
class Hyperrectangle:
    def __init__(self, low, high,npy=True):
        if npy:
            self.low = np.array(low, dtype=np.float32)
            self.high = np.array(high, dtype=np.float32)
        else:
            self.low = torch.tensor(low, dtype=torch.float32).to(device)
            self.high = torch.tensor(high, dtype=torch.float32).to(device)
        self.dim = len(low)
        self.npy = npy
    
    def contains(self, point):
        """Check if a point is inside the hyperrectangle."""
        assert self.npy
        low = np.array(self.low, dtype=np.float32)
        high = np.array(self.high, dtype=np.float32)
        return np.all(point >= low) and np.all(point <= high)
    
    def not_contains(self, point):
        """Check if a point is outside the hyperrectangle."""
        return not self.contains(point)
        
    def vertices_list(self):
        """Generate all vertices of the hyperrectangle"""
        n = len(self.low)
        vertices = []
        
        # Generate all combinations of low and high values
        for i in range(2**n):
            vertex = torch.zeros_like(self.low)
            for j in range(n):
                if (i >> j) & 1:
                    vertex[j] = self.high[j]
                else:
                    vertex[j] = self.low[j]
            vertices.append(vertex)
            
        return vertices

# 2D arm model
class PlanarArm:
    
    def __init__(self):
        self.state_dim = 2  # theta1, theta2
        self.control_dim = 2  # w1, w2
        
        self.l1 = 1.0
        self.l2 = 1.0
    
    def fx(self, x):
        return np.zeros((self.state_dim, 1))

    def gx(self, x):
        return np.eye(2)
    
    def dynamics(self, x, u, npy=True):
        """Compute the dynamics for Dubins car model."""
        if npy:
            x_dot = self.fx(x) + self.gx(x) @ u.reshape(-1, 1)
            x_dot = x_dot.reshape(-1)
        else:
            # if not torch.is_tensor(x): x = torch.tensor(x)
            # x_dot = torch.zeros_like(x)
            # x_dot[0] = u[0] * torch.cos(x[2])
            # x_dot[1] = u[0] * torch.sin(x[2])
            # x_dot[2] = u[1]
            
            if not torch.is_tensor(x): x = torch.tensor(x).to(device)
            fx = torch.from_numpy(self.fx(x)).to(device)
            gx = torch.from_numpy(self.gx(x)).to(device)
            u = torch.tensor(u, dtype=torch.float64).to(device).reshape(-1, 1)
            x_dot = fx + gx @ u
            x_dot = x_dot.reshape(-1)
        
        return x_dot
    
    def jacobian(self, x, u):
        """Compute the Jacobian of dynamics with respect to state and control"""
        if not torch.is_tensor(x):
            x = torch.tensor(x).unsqueeze(0)
        if not torch.is_tensor(u):
            u = torch.tensor(u).unsqueeze(0)
        batch_size = x.shape[0]
        A = torch.zeros(batch_size, self.state_dim, self.state_dim, device=x.device)
        B = torch.zeros(batch_size, self.state_dim, self.control_dim, device=x.device)
        
        # State Jacobian (A)
        # all zero
        
        # Control Jacobian (B)
        # identity
        B[:, 0, 0] = 1.0
        B[:, 1, 1] = 1.0
        
        return A, B
    
    def discrete_dynamics(self, x, u, t, dt, method='rk4'):
        """Compute the discrete dynamics using Runge-Kutta 4."""
        if method == 'rk4':
            # RK4 integration
            k1 = self.dynamics(x, u)
            k2 = self.dynamics(x + dt/2 * k1, u)
            k3 = self.dynamics(x + dt/2 * k2, u)
            k4 = self.dynamics(x + dt * k3, u)
            
            x_next = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            return x_next
        else:
            # Use scipy's ODE solver
            def f(t, x):
                return self.dynamics(x, u)
            
            sol = solve_ivp(f, [0, dt], x, method='RK45', rtol=1e-8, atol=1e-8)
            return sol.y[:, -1]

def random_point_in_hyperrectangle(hyperrectangle):
    """Generate a random point inside a hyperrectangle and outside a non-admissible area if specified."""
    dimensions = hyperrectangle.dim
    random_point = np.zeros(dimensions)
    
    for i in range(dimensions):
        random_point[i] = random.random() * (hyperrectangle.high[i] - hyperrectangle.low[i]) + hyperrectangle.low[i]
    
    return random_point

def pd_control(x, x_goal, v, Kp=1.0, Kd=0.01):
    
    error = x_goal - x
    error_dot = -v
    
    u = Kp * error + Kd * error_dot
    
    return u

def random_move_obstacle(x_obs, X_obs, U_obs, dt):
    
    v_obs = random_point_in_hyperrectangle(U_obs)
    x_obs = x_obs + v_obs * dt
    x_obs = np.clip(x_obs, X_obs.low, X_obs.high)
    
    return x_obs, v_obs

def dist_pt_to_line(pt, line_start, line_end):
    """Calculate the distance from a point to a line segment defined by two endpoints."""
    line_vec = line_end - line_start
    pt_vec = pt - line_start
    
    # Project point onto the line
    line_len_sq = np.dot(line_vec, line_vec)
    if line_len_sq == 0:
        return np.linalg.norm(pt - line_start)  # Line segment is a point
    
    projection = np.dot(pt_vec, line_vec) / line_len_sq
    projection = np.clip(projection, 0, 1)  # Clamp to the segment
    
    closest_point = line_start + projection * line_vec
    return np.linalg.norm(pt - closest_point)

def phi_whole_arm(x, arm, x_obs, dmin):
    '''
        Safety spec for planar arm.
        Used to label state safety.
        Used to derive phi for safe policy.
    '''
    
    theta1 = x[0]
    theta2 = x[1]
    l1 = arm.l1
    l2 = arm.l2
    
    p1 = np.array([0, 0])
    p2 = np.array([l1 * np.cos(theta1),
                   l1 * np.sin(theta1)])
    p3 = np.array([l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2),
                   l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)])
    
    d1 = dist_pt_to_line(x_obs, p1, p2)
    d2 = dist_pt_to_line(x_obs, p2, p3)
    
    phi = np.zeros(2)
    phi[0] = dmin - d1
    phi[1] = dmin - d2
    
    return phi

def generate_obstacle_location(X_obs, x_arm, arm, dmin, phi0):
    """Generate a random obstacle location that is initially safe."""
    for _ in range(50000):
        x_obs = random_point_in_hyperrectangle(X_obs)
        # Check if the obstacle is in a safe location
        if np.all(phi0(x_arm, arm, x_obs, dmin) <= 0):
            return x_obs
    
    return None

def compute_d_phi_d_x(x, arm, x_obs, dmin):
    
    delta = 1e-6
    grad = np.zeros((2, arm.state_dim))
    for i in range(arm.state_dim):
        dx = np.zeros(arm.state_dim)
        dx[i] = delta
        phi_plus = phi_whole_arm(x + dx, arm, x_obs, dmin)
        phi_minus = phi_whole_arm(x - dx, arm, x_obs, dmin)
        grad[:, i] = (phi_plus - phi_minus) / (2 * delta)
        
    return grad

def qp_solver(u_ref, Q_u, Lg, Lf, phi, alpha, U : Hyperrectangle = None, abs_=1e-4, eps_=1e-4):
    
    n = u_ref.shape[0]
    m = Lf.shape[0]
    
    Q = sparse.csc_matrix(Q_u)
    q = -(Q_u.T @ u_ref.reshape(-1, 1)).reshape(-1)
    
    C_upper = sparse.csc_matrix(Lg)
    if U:
        C_lower = sparse.eye(n)
        Amat = sparse.vstack([C_upper, C_lower]).tocsc()
        u_low = np.array([U.low[i] for i in range(n)])
        u_high = np.array([U.high[i] for i in range(n)])
        l = np.concatenate([
            -np.inf * np.ones(m),
            u_low
        ])
        u = np.concatenate([
            -alpha * phi.reshape(-1) - Lf.reshape(-1),
            u_high
        ])
    else:
        Amat = C_upper
        l = -np.inf * np.ones(m)
        u = -alpha * phi.reshape(-1) - Lf.reshape(-1)
    
    # Solve the quadratic program
    prob = osqp.OSQP()
    prob.setup(Q, q, Amat, l, u,
                rho=0.1, adaptive_rho=True,
                eps_abs=abs_, eps_rel=eps_,
                max_iter=20000,
                polish=False, verbose=False)
    result = prob.solve()
    
    if result.info.status != 'solved':
        # assert U is not None, "QP solver failed to find a solution with unbounded control"
        return u_ref
    else:
        return result.x.reshape(-1)

def safe_control(u_ref, U, x, arm, x_obs, dmin, alpha):
    
    phi = phi_whole_arm(x, arm, x_obs, dmin)
    d_phi_d_x = compute_d_phi_d_x(x, arm, x_obs, dmin)
    Lf = d_phi_d_x @ arm.fx(x)
    Lg = d_phi_d_x @ arm.gx(x)
    
    Q_u = np.eye(arm.control_dim)  # Control cost
    
    u = qp_solver(u_ref=u_ref, Q_u=Q_u, Lg=Lg, Lf=Lf, phi=phi, alpha=alpha, U=U)
    
    if u is None:
        u = qp_solver(u_ref=u_ref, Q_u=Q_u, Lg=Lg, Lf=Lf, phi=phi, alpha=alpha, U=None)
    
    assert u is not None, "Safe control could not be computed"
    
    # now clip to the control limits
    u = np.clip(u, U.low, U.high)
    
    return u, phi

class Obstacle:
    def __init__(self, x, X, U, dt, T):
        """
        Initialize the obstacle.
        Args:
            x (np.ndarray): Initial position of the obstacle.
            X (Hyperrectangle): State space of the obstacle.
            U (Hyperrectangle): Control space of the obstacle.
            dt (float): Discrete time step.
            T (float): Frequency of changing velocity (in seconds).
        """
        self.x = np.array(x, dtype=np.float32)
        self.X = X
        self.U = U
        self.dt = dt
        self.T = T
        self.time_elapsed = 0.0
        self.v = random_point_in_hyperrectangle(U)  # Initial velocity

    def step(self):
        """
        Move the obstacle by one time step.
        """
        # Resample velocity every T seconds
        if self.time_elapsed >= self.T:
            self.v = random_point_in_hyperrectangle(self.U)
            self.time_elapsed = 0.0

        # Update position
        self.x += self.v * self.dt

        # Ensure the obstacle stays within its state space
        self.x = np.clip(self.x, self.X.low, self.X.high)
        
        self.time_elapsed += self.dt
        
        return self.x, self.v

def generate_traj(
        x_0, x_goal, U, arm, kp, kd,
        x_obs_0, X_obs, U_obs, T_obs,
        dmin, alpha,
        dt, traj_len_T
    ):
    """Generate a reference trajectory starting from x_0."""
    
    traj_len = int(np.floor(traj_len_T / dt))
    x_traj = []
    u_traj = []
    x = copy.deepcopy(x_0)
    u = np.zeros(arm.control_dim)
    
    x_obs_traj = []
    u_obs_traj = []
    x_obs = copy.deepcopy(x_obs_0)
    obstacle = Obstacle(x_obs, X_obs, U_obs, dt, T_obs)
    
    phi_traj = []
    
    for i in range(traj_len):
        
        # Goal tracking action
        u_ref = pd_control(x, x_goal, u, Kp=kp, Kd=kd)
        
        # Safety filter
        u, phi = safe_control(u_ref=u_ref, U=U, x=x, arm=arm, x_obs=x_obs, dmin=dmin, alpha=alpha)
        # u[:] = u_ref[:]
        
        # Update arm state using discrete dynamics
        x_next = arm.discrete_dynamics(x, u, 0.0, dt)
        
        # Sample obstacle motion
        x_obs_next, u_obs = obstacle.step()
        
        # Record trajectory
        x_traj.append(x)
        u_traj.append(u)
        x_obs_traj.append(x_obs)
        u_obs_traj.append(u_obs)
        phi_traj.append(phi)
        
        # forward
        x = copy.deepcopy(x_next)
        x_obs = copy.deepcopy(x_obs_next)
    
    return x_traj, u_traj, x_obs_traj, u_obs_traj, phi_traj

def generate_random_trajs(dmodel, num_traj, dt, traj_len_T, X_init, X_goal, U, X_obs, U_obs, T_obs, kp, kd, dmin, alpha):
    """Generate multiple random trajectories."""
    x_goals = []  # Record x_goal for each trajectory
    x_trajs = []
    u_trajs = []
    x_obs_trajs = []
    u_obs_trajs = []
    phi_trajs = []
    
    for i in tqdm(range(num_traj), desc="Generating trajectories"):
        x_0 = random_point_in_hyperrectangle(X_init)
        x_goal = random_point_in_hyperrectangle(X_goal)
        x_obs_0 = generate_obstacle_location(
            X_obs=X_obs,
            x_arm=x_0,
            arm=dmodel,
            dmin=dmin,
            phi0=phi_whole_arm
        )
        
        x_traj, u_traj, x_obs_traj, u_obs_traj, phi_traj = generate_traj(
            x_0=x_0, x_goal=x_goal, U=U, arm=dmodel, kp=kp, kd=kd,
            x_obs_0=x_obs_0, X_obs=X_obs, U_obs=U_obs, T_obs=T_obs,
            dmin=dmin, alpha=alpha,
            dt=dt, traj_len_T=traj_len_T
        )
        x_goals.append(x_goal)  # Save x_goal
        x_trajs.append(x_traj)
        u_trajs.append(u_traj)
        x_obs_trajs.append(x_obs_traj)
        u_obs_trajs.append(u_obs_traj)
        phi_trajs.append(phi_traj)
    
    return x_goals, x_trajs, u_trajs, x_obs_trajs, u_obs_trajs, phi_trajs

def build_dataset(name, x_trajs, u_trajs, x_obs_trajs, u_obs_trajs, phi_trajs, FI_steps, train_data_ratio, root):
    """
    Build a dataset from the generated trajectories.
    This function labels the data based on the safety specification phi,
    and splits it into training and test sets.
    Args:
        name (str): Name of the dataset.
        x_trajs (list): List of state trajectories.
        u_trajs (list): List of control trajectories.
        x_obs_trajs (list): List of obstacle state trajectories.
        u_obs_trajs (list): List of obstacle control trajectories.
        phi_trajs (list): List of safety specification values for each trajectory.
        FI_steps (int): Number of steps to consider for FI.
        train_data_ratio (float): Ratio of training data to total data.
    Returns:
        train_data (list): List of training data tuples (x, u, x_obs, u_obs, y).
        test_data (list): List of test data tuples (x, u, x_obs, u_obs, y).
    """
    data = []
    
    # label the data
    phi_arr = np.array([np.array(traj) for traj in phi_trajs], dtype=object)
    safe_trajs = (phi_arr <= 0).all(axis=-1) # size: (num_traj, traj_len)
    
    X_data_FI = []
    U_data_FI = []
    Xobs_data_FI = []
    Uobs_data_FI = []
    Y_data_FI = []
    
    X_data_nonFI = []
    U_data_nonFI = []
    Xobs_data_nonFI = []
    Uobs_data_nonFI = []
    Y_data_nonFI = []
    
    for i in range(len(x_trajs)):
        safe_traj = safe_trajs[i]
        
        # --------------------------- Collect nonFI states --------------------------- #
        unsafe_ids = np.where(~safe_traj)[0].tolist()
        if len(unsafe_ids) > 0:
            X_data_nonFI.extend([x_trajs[i][j] for j in unsafe_ids])
            U_data_nonFI.extend([u_trajs[i][j] for j in unsafe_ids])
            Xobs_data_nonFI.extend([x_obs_trajs[i][j] for j in unsafe_ids])
            Uobs_data_nonFI.extend([u_obs_trajs[i][j] for j in unsafe_ids])
            Y_data_nonFI.extend([safe_trajs[i][j] for j in unsafe_ids])

        # todo compare including safe states leading unsafe states
        
        # ----------------------------- Collect FI states ---------------------------- #
        if len(unsafe_ids) > 0:
            # If there are unsafe states, we only take the safe part of the trajectory
            last_FI_id = unsafe_ids[0] - 1 - FI_steps
        else:
            # If no unsafe states, whole traj but the last FI_steps are FI
            last_FI_id = len(safe_traj) - 1 - FI_steps
            
        if last_FI_id >= 0:
            X_data_FI.extend(x_trajs[i][:last_FI_id + 1])
            U_data_FI.extend(u_trajs[i][:last_FI_id + 1])
            Xobs_data_FI.extend(x_obs_trajs[i][:last_FI_id + 1])
            Uobs_data_FI.extend(u_obs_trajs[i][:last_FI_id + 1])
            Y_data_FI.extend(safe_traj[:last_FI_id + 1])
    
    num_FI = len(X_data_FI)
    num_nonFI = len(X_data_nonFI)
    print(f"Number of FI samples: {num_FI}, Number of non-FI samples: {num_nonFI}")
    
    # Combine into list of tuples
    data_FI = list(zip(X_data_FI, U_data_FI, Xobs_data_FI, Uobs_data_FI, Y_data_FI))
    data_nonFI = list(zip(X_data_nonFI, U_data_nonFI, Xobs_data_nonFI, Uobs_data_nonFI, Y_data_nonFI))
    
    # split into training and test sets
    random.shuffle(data_FI)
    random.shuffle(data_nonFI)
    
    # Combine training and testing data
    num_FI_train = int(num_FI * train_data_ratio)
    num_nonFI_train = int(num_nonFI * train_data_ratio)
    train_data = data_FI[:num_FI_train] + data_nonFI[:num_nonFI_train]
    test_data = data_FI[num_FI_train:] + data_nonFI[num_nonFI_train:]
    
    # Shuffle the training and test data
    random.shuffle(train_data)
    random.shuffle(test_data)
    
    # Save to files
    with open(os.path.join(root, f"{name}_train_data.pkl"), "wb") as f:
        pickle.dump(train_data, f)
    
    with open(os.path.join(root, f"{name}_test_data.pkl"), "wb") as f:
        pickle.dump(test_data, f)
    
    print(f"Saved {len(train_data)} training samples and {len(test_data)} test samples")
    
    return train_data, test_data

def draw_arm(ax, arm, x, phi, color='blue'):
    """Draw the arm given its state x."""
    theta1, theta2 = x
    l1, l2 = arm.l1, arm.l2

    p1 = np.array([0, 0])
    p2 = np.array([l1 * np.cos(theta1), l1 * np.sin(theta1)])
    p3 = np.array([l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2),
                    l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)])

    # Highlight links if phi is larger than zero
    link1_color = 'orange' if phi[0] > 0 else color
    link2_color = 'orange' if phi[1] > 0 else color

    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=link1_color, linewidth=2)
    ax.plot([p2[0], p3[0]], [p2[1], p3[1]], color=link2_color, linewidth=2)
    ax.scatter([p1[0], p2[0], p3[0]], [p1[1], p2[1], p3[1]], color=color)

def draw_obstacle(ax, x_obs, color='red'):
    """Draw the obstacle as a point."""
    ax.scatter(x_obs[0], x_obs[1], color=color, s=100)

def draw_goal(ax, arm, x_goal, color='green'):
    """Draw the goal position of the arm."""
    theta1, theta2 = x_goal
    l1, l2 = arm.l1, arm.l2

    p1 = np.array([0, 0])
    p2 = np.array([l1 * np.cos(theta1), l1 * np.sin(theta1)])
    p3 = np.array([l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2),
                    l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)])

    ax.scatter([p3[0]], [p3[1]], color=color, s=100, label="Goal")

def animate_trajectory(x_traj, x_obs_traj, phi_traj, x_goal, arm, save_path, traj_id):
    """Create an animation for a single trajectory."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-arm.l1 - arm.l2 - 0.5, arm.l1 + arm.l2 + 0.5)
    ax.set_ylim(-arm.l1 - arm.l2 - 0.5, arm.l1 + arm.l2 + 0.5)
    ax.set_aspect('equal')
    ax.set_title(f"Trajectory {traj_id}")

    def update(frame):
        ax.clear()
        ax.set_xlim(-arm.l1 - arm.l2 - 0.5, arm.l1 + arm.l2 + 0.5)
        ax.set_ylim(-arm.l1 - arm.l2 - 0.5, arm.l1 + arm.l2 + 0.5)
        ax.set_aspect('equal')
        draw_arm(ax, arm, x_traj[frame], phi_traj[frame])
        draw_obstacle(ax, x_obs_traj[frame])
        draw_goal(ax, arm, x_goal)

    ani = animation.FuncAnimation(fig, update, frames=len(x_traj), interval=100)
    ani.save(os.path.join(save_path, f"traj_{traj_id}.gif"), writer='imagemagick')
    plt.close(fig)

def plot_trajectories(x_goals, x_trajs, u_trajs, x_obs_trajs, u_obs_trajs, phi_trajs, arm, num_to_save, save_path):
    '''
    Plot the trajectories of the arm and obstacles.
    Plot arm, obstacle, and goal, and save the animation in gif.
    '''

    # Randomly pick trajectories to save
    selected_indices = random.sample(range(len(x_trajs)), num_to_save)
    for idx in selected_indices:
        animate_trajectory(x_trajs[idx], x_obs_trajs[idx], phi_trajs[idx], x_goals[idx], arm, save_path, idx)

# Main execution
if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(24)
    random.seed(24)
    
    # Paths for saving data and plots
    exp_name = "batch2"
    data_path = os.path.join(exp_name, "data")
    plot_path = os.path.join(exp_name, "viz")
    num_to_save = 10  # Number of trajectories to save for visualization

    # Create directories if they don't exist
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(plot_path, exist_ok=True)
    
    # -------------------- Hyperparameters for tuning --------------------
    num_traj = int(1e3)          # Number of trajectories to generate
    dt = 0.1               # Time step for simulation
    traj_len_T = 10.0      # Total trajectory length in seconds
    T_obs = 2.0            # Time interval for obstacle velocity change
    kp = 1.0               # Proportional gain for PD control
    kd = 0.01              # Derivative gain for PD control
    dmin = 0.2             # Safety margin
    alpha = 10.0          # Safety filter parameter
    FI_steps = 50          # Number of steps to consider for FI
    train_data_ratio = 0.6 # Ratio of training data to total data

    # -------------------- Define dynamic model and spaces --------------------
    # Create the dynamic model
    arm = PlanarArm()
    
    # Define the state and control spaces
    X_init = Hyperrectangle(low=[0, 0], high=[np.pi/2.0, np.pi/2.0])
    X_goal = Hyperrectangle(low=[0, 0], high=[np.pi/2.0, np.pi/2.0])
    U = Hyperrectangle(low=[-1, -1], high=[1, 1])
    
    # Obstacle state and control spaces
    X_obs = Hyperrectangle(low=[0, 0], high=[arm.l1+arm.l2, arm.l1+arm.l2])
    U_obs = Hyperrectangle(low=[-0.5, -0.5], high=[0.5, 0.5])
    
    # -------------------- Generate trajectories --------------------
    print("Generating random trajectories...")
    x_goals, x_trajs, u_trajs, x_obs_trajs, u_obs_trajs, phi_trajs = generate_random_trajs(
        dmodel     = arm,
        num_traj   = num_traj,
        dt         = dt,
        traj_len_T = traj_len_T,
        X_init     = X_init,
        X_goal     = X_goal,
        U          = U,
        X_obs      = X_obs,
        U_obs      = U_obs,
        T_obs      = T_obs,
        kp         = kp,
        kd         = kd,
        dmin       = dmin,
        alpha      = alpha
    )
    print(f"Generated {len(x_trajs)} trajectories")
    
    # -------------------- Plot some trajectories --------------------
    plot_trajectories(x_goals, x_trajs, u_trajs, x_obs_trajs, u_obs_trajs, phi_trajs, arm, num_to_save, plot_path)
    
    # -------------------- Build and save the dataset --------------------
    print("Building dataset...")
    train_data, test_data = build_dataset(
        name="planar_arm",
        x_trajs=x_trajs,
        u_trajs=u_trajs,
        x_obs_trajs=x_obs_trajs,
        u_obs_trajs=u_obs_trajs,
        phi_trajs=phi_trajs,
        FI_steps=FI_steps,
        train_data_ratio=train_data_ratio,
        root=data_path
    )
    
    print("Data collection complete!")