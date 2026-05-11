import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Setup
torch.manual_seed(364)
np.random.seed(364)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dimension = 2
nb_steps = 200
T = 10.0  
dt = T / nb_steps
batch_size = 128
q_bound = 1.0     
pi_bound = 10.0   

c = torch.eye(dimension) * 0.25 
Sigma_fixed = torch.eye(dimension) * 0.25 
inv_Sigma = torch.inverse(Sigma_fixed)
B_sym = 0.5 * torch.matmul(c, inv_Sigma)

# Architecture
def get_ffnn(input_size, output_size, nn_desc):
    layers = []
    curr = input_size
    for h_dim, activ in nn_desc:
        layers.append(nn.Linear(curr, h_dim))
        layers.append(nn.LeakyReLU())
        curr = h_dim
    layers.append(nn.Linear(curr, output_size))
    return nn.Sequential(*layers)

def get_skew_symmetric(params, d, bs):
    Q = torch.zeros(bs, d, d, device=device)
    indices = torch.triu_indices(d, d, offset=1, device=device)
    Q[:, indices[0], indices[1]] = params
    return Q - Q.transpose(1, 2)

def sample_langevin(bs, d, inv_sig, steps=50):
    X = torch.randn(bs, d, device=device)
    for _ in range(steps):
        # Force derived from Fokker-Planck for N(0, Sigma)
        grad = -0.5 * (X @ inv_sig.T)
        X = X + grad * 0.1 + torch.randn_like(X) * np.sqrt(0.1)
    return X

# RobustModel
class RobustModel(nn.Module):
    def __init__(self, dimension, config):
        super().__init__()
        self.dimension = dimension
        self.generator = get_ffnn(dimension + 1, dimension, config)
        self.n_skew = int(dimension * (dimension - 1) / 2)

        # Nature's learnable Q parameter
        self.adv_param = nn.Parameter(torch.zeros(self.n_skew))

    def forward(self, dWs, which="gen", q_override=None):
        bs = dWs.shape[0]

        current_X = sample_langevin(bs, self.dimension, inv_Sigma.to(device))
        
        # Build Adversarial Drift B satisfying Fokker-Planck
        if q_override is None:
            q_params = q_bound * torch.tanh(self.adv_param).repeat(bs, 1)
        else:
            q_params = torch.full(
                (bs, self.n_skew),
                float(q_override),
                device=device
            )

        Q = get_skew_symmetric(q_params, self.dimension, bs)

        B = B_sym.to(device) + torch.bmm(
            Q,
            inv_Sigma.to(device).repeat(bs, 1, 1)
        )
        
        log_wealth = torch.zeros(bs, 1, device=device)
        
        for i in range(dWs.shape[2]):
            # Features: current state X and normalized time
            t_norm = torch.ones(bs, 1, device=device) * (i * dt / T)
            input_feat = torch.cat([current_X, t_norm], dim=1)
            
            # Agent strategy pi(X)
            pi = pi_bound * torch.tanh(self.generator(input_feat))
            
            # Dynamics dX = -B X dt + sqrt(c) dW
            mu_x = -torch.bmm(B, current_X.unsqueeze(2)).squeeze(2)

            noise = torch.matmul(
                torch.linalg.cholesky(c).to(device),
                dWs[:, :, i].unsqueeze(2)
            ).squeeze(2)

            dX = mu_x * dt + noise
            
            # Asymptotic Growth Calculation
            drift_gain = torch.sum(pi * dX, dim=1, keepdim=True)
            risk_penalty = 0.5 * torch.sum((pi @ c.to(device)) * pi, dim=1, keepdim=True) * dt

            log_wealth += drift_gain - risk_penalty
            current_X = current_X + dX

        avg_growth = torch.mean(log_wealth) / T

        return -avg_growth if which == "gen" else avg_growth

# Training
config = ((64, 'leaky_relu'), (64, 'leaky_relu'))
model = RobustModel(dimension, config).to(device)

opt_gen = torch.optim.Adam(model.generator.parameters(), lr=1e-3)
opt_adv = torch.optim.Adam([model.adv_param], lr=1e-3)

history = []

for epoch in range(401):
    dWs = torch.randn(batch_size, dimension, nb_steps, device=device) * np.sqrt(dt)
    
    # Nature tries to find the worst-case B
    opt_adv.zero_grad()
    loss_adv = model(dWs, which="adv")
    loss_adv.backward()
    opt_adv.step()
    
    # Agent finds best response pi
    opt_gen.zero_grad()
    loss_gen = model(dWs, which="gen")
    loss_gen.backward()
    opt_gen.step()
    
    current_val = -loss_gen.item()
    history.append(current_val)

    if epoch % 50 == 0:
        print(f"Epoch {epoch:03d} | Robust Growth Rate: {current_val:.6f}")

# Robustness Check
@torch.no_grad()
def estimate_growth_for_q(model, q_value, n_paths=5000, eval_batch_size=500):
    model.eval()

    total_growth = 0.0
    total_paths = 0
    remaining = n_paths

    while remaining > 0:
        bs = min(eval_batch_size, remaining)

        dWs = torch.randn(bs, dimension, nb_steps, device=device) * np.sqrt(dt)

        loss = model(dWs, which="gen", q_override=q_value)
        growth = -loss.item()

        total_growth += growth * bs
        total_paths += bs
        remaining -= bs

    return total_growth / total_paths

qs = np.linspace(-q_bound, q_bound, 41)
growths = []

for q in qs:
    g = estimate_growth_for_q(model, q, n_paths=5000, eval_batch_size=500)
    growths.append(g)

growths = np.array(growths)

worst_idx = int(np.argmin(growths))
worst_q = qs[worst_idx]
worst_growth = growths[worst_idx]

theoretical_growth = 0.125 * torch.trace(c.to(device) @ inv_Sigma.to(device)).item()

print("\n========== Robustness Check ==========")
print(f"Theoretical growth: {theoretical_growth:.6f}")
print(f"Worst q on grid: {worst_q:.4f}")
print(f"Worst-case learned growth: {worst_growth:.6f}")
print(f"Best-case learned growth: {float(np.max(growths)):.6f}")

# Robustness curve
plt.figure(figsize=(7, 4))
plt.plot(qs, growths, label="Learned Agent")
plt.axhline(theoretical_growth, color="crimson", linestyle="--", label="Theory")
plt.axvline(worst_q, color="black", linestyle=":", label=f"Worst q={worst_q:.2f}")
plt.xlabel("Adversary q")
plt.ylabel("Estimated growth rate")
plt.title("Growth against admissible adversarial q")
plt.legend()
plt.grid(True)
plt.show()

# Plotting & Results
def plot_robust_results(model, history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Growth History
    ax1.plot(history, color='navy', label='ML Growth Rate')
    ax1.axhline(y=theoretical_growth, color='crimson', linestyle='--', label='Part I Theoretical')
    ax1.set_title("Growth Rate Invariance Convergence")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Growth Rate $g$")
    ax1.legend()

    # Strategy Field pi(X)
    x = np.linspace(-1, 1, 15)
    y = np.linspace(-1, 1, 15)
    X, Y = np.meshgrid(x, y)

    grid = torch.tensor(
        np.stack([X.flatten(), Y.flatten()], axis=1),
        dtype=torch.float32
    ).to(device)

    t_in = torch.zeros(grid.shape[0], 1).to(device)
    
    model.eval()
    with torch.no_grad():
        pi_out = (
            pi_bound * torch.tanh(model.generator(torch.cat([grid, t_in], dim=1)))
        ).cpu().numpy()
    
    ax2.quiver(X, Y, pi_out[:, 0], pi_out[:, 1], color='darkcyan')
    ax2.set_title(r"Learned Strategy Field $\pi(X)$")
    ax2.set_xlabel(r"$X_1$")
    ax2.set_ylabel(r"$X_2$")

    plt.show()

plot_robust_results(model, history)