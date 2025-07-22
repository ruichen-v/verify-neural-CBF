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
        return np.ones((self.state_dim, self.control_dim))
    
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
            u = torch.tensor(u, dtype=torch.float32).to(device).reshape(-1, 1)
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
                polish=True, verbose=False)
    result = prob.solve()
    
    if result.info.status_val != osqp.constant.OSQP_SOLVED:
        assert U is not None, "QP solver failed to find a solution with unbounded control"
        return None
    else:
        return result.x.reshape(-1, 1)

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

def generate_traj(
        x_0, x_goal, U, arm, kp, kd,
        x_obs_0, X_obs, U_obs, dmin,
        alpha,
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
    u_obs = np.zeros(2)
    
    phi_traj = []
    
    for i in range(traj_len):
        
        # Goal tracking action
        u_ref = pd_control(x, x_goal, u, Kp=kp, Kd=kd)
        
        # Safety filter
        u, phi = safe_control(u_ref=u_ref, U=U, x=x, arm=arm, x_obs=x_obs, dmin=dmin, alpha=alpha)
        
        # Update arm state using discrete dynamics
        x_next = arm.discrete_dynamics(x, u, 0.0, dt)
        
        # Sample obstacle motion
        x_obs_next, u_obs = random_move_obstacle(x_obs, X_obs, U_obs, dt)
        
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

def generate_random_trajs(dmodel, num_traj, dt, traj_len_T, X_init, X_goal, U, X_obs, U_obs, kp, kd, dmin, alpha):
    """Generate multiple random trajectories."""
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
            x_obs_0=x_obs_0, X_obs=X_obs, U_obs=U_obs, dmin=dmin,
            alpha=alpha,
            dt=dt, traj_len_T=traj_len_T
        )
        x_trajs.append(x_traj)
        u_trajs.append(u_traj)
        x_obs_trajs.append(x_obs_traj)
        u_obs_trajs.append(u_obs_traj)
        phi_trajs.append(phi_traj)
    
    return x_trajs, u_trajs, x_obs_trajs, u_obs_trajs, phi_trajs

def build_dataset(name, x_trajs, u_trajs, x_obs_trajs, u_obs_trajs, phi_trajs, FI_steps, train_data_ratio):
    """Build and save the dataset for training and testing."""
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
        unsafe_ids = np.where(~safe_traj)[0]
        if len(unsafe_ids) > 0:
            X_data_nonFI.extend(x_trajs[i][unsafe_ids])
            U_data_nonFI.extend(u_trajs[i][unsafe_ids])
            Xobs_data_nonFI.extend(x_obs_trajs[i][unsafe_ids])
            Uobs_data_nonFI.extend(u_obs_trajs[i][unsafe_ids])
            Y_data_nonFI.extend(safe_trajs[i][unsafe_ids])
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
    rand_ids_FI = np.random.permutation(num_FI)
    data_FI_shuffled = data_FI[rand_ids_FI]
    
    rand_ids_nonFI = np.random.permutation(num_nonFI)
    data_nonFI_shuffled = data_nonFI[rand_ids_nonFI]
    
    # Combine training and testing data
    num_FI_train = int(num_FI * train_data_ratio)
    num_nonFI_train = int(num_nonFI * train_data_ratio)
    train_data = data_FI_shuffled[:num_FI_train] + data_nonFI_shuffled[:num_nonFI_train]
    test_data = data_FI_shuffled[num_FI_train:] + data_nonFI_shuffled[num_nonFI_train:]
    
    # Shuffle the training and test data
    random.shuffle(train_data)
    random.shuffle(test_data)
    
    # Save to files
    with open(f"data/{name}_training_data.pkl", "wb") as f:
        pickle.dump(train_data, f)
    
    with open(f"data/{name}_test_data.pkl", "wb") as f:
        pickle.dump(test_data, f)
    
    print(f"Saved {len(train_data)} training samples and {len(test_data)} test samples")
    
    return train_data, test_data

def plot_trajectories(Xrefs, title="Dubins Car Trajectories", max_traj=np.inf):
    """Plot some of the generated trajectories."""
    plt.figure(figsize=(10, 8))
    
    # Plot a subset of trajectories to avoid cluttering
    for i in range(min(len(Xrefs), max_traj)):
        x_coords = [x[0] for x in Xrefs[i]]
        y_coords = [x[1] for x in Xrefs[i]]
        plt.plot(x_coords, y_coords, 'b-', alpha=0.3)
        
        # Mark the starting point
        plt.plot(x_coords[0], y_coords[0], 'go')
    
    # Plot the unsafe region
    unsafe_x = [X_unsafe.low[0], X_unsafe.high[0], X_unsafe.high[0], X_unsafe.low[0], X_unsafe.low[0]]
    unsafe_y = [X_unsafe.low[1], X_unsafe.low[1], X_unsafe.high[1], X_unsafe.high[1], X_unsafe.low[1]]
    plt.plot(unsafe_x, unsafe_y, 'r-', linewidth=2)
    
    # Plot the domain
    domain_x = [X.low[0], X.high[0], X.high[0], X.low[0], X.low[0]]
    domain_y = [X.low[1], X.low[1], X.high[1], X.high[1], X.low[1]]
    plt.plot(domain_x, domain_y, 'k-', linewidth=2)
    
    plt.title(title)
    plt.xlabel('x position')
    plt.ylabel('y position')
    plt.grid(True)
    plt.savefig(f"{title.lower().replace(' ', '_')}.png")
    plt.show()

# Main execution
if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Create the dynamic model
    arm = PlanarArm()
    
    # Define the state and control spaces
    X_init = Hyperrectangle(low=[0, 0], high=[np.pi/2.0, np.pi/2.0])
    X_goal = Hyperrectangle(low=[0, 0], high=[np.pi/2.0, np.pi/2.0])
    U = Hyperrectangle(low=[-1, -1], high=[1, 1])
    
    # obstacle state and control
    X_obs = Hyperrectangle(low=[0, 0], high=[arm.l1+arm.l2, arm.l1+arm.l2])
    U_obs = Hyperrectangle(low=[-0.5, -0.5], high=[0.5, 0.5])
    
    # Safety margin
    dmin = 0.2
    
    # Generate trajectories
    print("Generating random trajectories...")
    x_trajs, u_trajs, x_obs_trajs, u_obs_trajs, phi_trajs = generate_random_trajs(
        dmodel     = arm,
        num_traj   = 1000,
        dt         = 0.1,
        traj_len_T = 10.0,
        X_init     = X_init,
        X_goal     = X_goal,
        U          = U,
        X_obs      = X_obs,
        U_obs      = U_obs,
        kp         = 1.0,
        kd         = 0.01,
        dmin       = dmin,
        alpha      = 0.01
    )
    
    # Plot some trajectories
    # plot_trajectories(Xrefs)
    
    # Build and save the dataset
    print("Building dataset...")
    train_data, test_data = build_dataset(
        name="planar_arm",
        x_trajs=x_trajs,
        u_trajs=u_trajs,
        x_obs_trajs=x_obs_trajs,
        u_obs_trajs=u_obs_trajs,
        phi_trajs=phi_trajs,
        FI_steps=50,  # Number of steps to consider for FI
        train_data_ratio=0.6  # Ratio of training data
    )
    
    print("Data collection complete!")
    print(f"Generated {len(x_trajs)} trajectories")
    print(f"Created {len(train_data)} training samples and {len(test_data)} test samples")