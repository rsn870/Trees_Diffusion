"""
Experiment 1: Implicit Tree Discovery via Diffusion
====================================================

Discovers hierarchical cluster structure by training a diffusion model
and analyzing merge times during forward diffusion process.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import make_blobs
import itertools
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Diffusion hyperparameters
T = 1.0  # Total diffusion time
N_STEPS = 100  # Number of discrete timesteps
dt = T / N_STEPS

# Noise schedule
BETAS = torch.linspace(0.0001, 0.02, N_STEPS).to(DEVICE)
ALPHAS = (1. - BETAS).to(DEVICE)
ALPHAS_CUMPROD = torch.cumprod(ALPHAS, axis=0).to(DEVICE)

# ============================================================================
# Dataset Generation
# ============================================================================

def get_dataset(name='4_corners', n_samples=2000):
    """Generate synthetic clustered datasets."""
    if name == '4_corners':
        centers = [[-2, -2], [-2, 2], [2, -2], [2, 2]]
        X, y = make_blobs(n_samples=n_samples, centers=centers, 
                         cluster_std=0.3, random_state=42)
    
    elif name == '9_grid':
        centers = [[-2,-2], [-2,0], [-2,2], 
                  [0,-2], [0,0], [0,2], 
                  [2,-2], [2,0], [2,2]]
        X, y = make_blobs(n_samples=n_samples, centers=centers, 
                         cluster_std=0.25, random_state=42)
    
    elif name == '8_gaussians':
        scale = 2.
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        centers = [(scale*np.cos(a), scale*np.sin(a)) for a in angles]
        
        X_list, y_list = [], []
        samples_per_cluster = n_samples // 8
        for i, (x_c, y_c) in enumerate(centers):
            points = np.random.randn(samples_per_cluster, 2) * 0.15 + [x_c, y_c]
            X_list.append(points)
            y_list.extend([i] * samples_per_cluster)
        
        X = np.concatenate(X_list)
        y = np.array(y_list)
    
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return torch.tensor(X, dtype=torch.float32), y

# ============================================================================
# Diffusion Model
# ============================================================================

class DiffusionBlock(nn.Module):
    """Residual block for diffusion model."""
    def __init__(self, nunits):
        super().__init__()
        self.linear = nn.Linear(nunits, nunits)
    
    def forward(self, x):
        return nn.functional.relu(self.linear(x))

class DiffusionModel(nn.Module):
    """Score-based diffusion model."""
    def __init__(self, nfeatures: int, nblocks: int = 4, nunits: int = 128):
        super().__init__()
        self.inblock = nn.Linear(nfeatures + 1, nunits)
        self.midblocks = nn.ModuleList([
            DiffusionBlock(nunits) for _ in range(nblocks)
        ])
        self.outblock = nn.Linear(nunits, nfeatures)
    
    def forward(self, x, t):
        """
        Args:
            x: Noised data (batch_size, nfeatures)
            t: Time values (batch_size,) in [0, 1]
        Returns:
            Predicted noise (batch_size, nfeatures)
        """
        t_reshaped = t.view(-1, 1)
        val = torch.hstack([x, t_reshaped])
        val = self.inblock(val)
        
        for midblock in self.midblocks:
            val = midblock(val)
        
        return self.outblock(val)

# ============================================================================
# Training
# ============================================================================

def analytical_forward_process(x0, t_idx):
    """Apply forward diffusion analytically."""
    noise = torch.randn_like(x0)
    sqrt_alpha_t = torch.sqrt(ALPHAS_CUMPROD[t_idx]).view(-1, 1)
    sqrt_one_minus_alpha_t = torch.sqrt(1. - ALPHAS_CUMPROD[t_idx]).view(-1, 1)
    
    x_t = sqrt_alpha_t * x0 + sqrt_one_minus_alpha_t * noise
    return x_t, noise

def train_model(X_data, nepochs=400, batch_size=2048, lr=1e-3):
    """Train diffusion model to predict added noise."""
    model = DiffusionModel(nfeatures=2, nblocks=4, nunits=128).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=nepochs
    )
    loss_fn = nn.MSELoss()
    
    batch_size = min(batch_size, X_data.shape[0])
    
    pbar = tqdm(range(nepochs), desc="Training Diffusion Model")
    for epoch in pbar:
        epoch_loss = 0
        indices = torch.randperm(X_data.shape[0])
        
        for i in range(0, len(X_data), batch_size):
            batch_indices = indices[i:i+batch_size]
            Xbatch = X_data[batch_indices].to(DEVICE)
            
            # Random timesteps
            timesteps_int = torch.randint(0, N_STEPS, (Xbatch.shape[0],), 
                                         device=DEVICE)
            
            # Forward process
            noised_batch, eps = analytical_forward_process(Xbatch, timesteps_int)
            
            # Normalized time [0, 1]
            t_float = timesteps_int.float() / (N_STEPS - 1)
            
            # Predict noise
            predicted_noise = model(noised_batch, t_float)
            
            # Loss
            loss = loss_fn(predicted_noise, eps)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / (len(X_data) / batch_size)
        pbar.set_postfix({"Loss": f"{avg_loss:.4f}"})
        scheduler.step()
    
    return model.eval()

# ============================================================================
# Tree Discovery via Forward SDE Simulation
# ============================================================================

@torch.no_grad()
def learned_forward_sde_step(model, x_t, t_idx):
    """Single step of learned forward SDE."""
    t_float = t_idx / (N_STEPS - 1)
    t_tensor = torch.full((x_t.shape[0],), t_float, device=DEVICE)
    
    beta_t = BETAS[t_idx]
    
    # Predict noise and compute score
    predicted_noise = model(x_t, t_tensor)
    score = -predicted_noise / torch.sqrt(1 - ALPHAS_CUMPROD[t_idx])
    
    # SDE dynamics: dx = drift*dt + diffusion*dW
    drift = -0.5 * beta_t * x_t - 0.5 * beta_t * score
    diffusion = torch.sqrt(beta_t) * torch.randn_like(x_t)
    
    x_t_plus_dt = x_t + drift * dt + diffusion
    return x_t_plus_dt

def discover_tree_structure(model, X_data, y_true):
    """
    Discover hierarchical structure by simulating forward diffusion
    and tracking cluster merge times.
    """
    print("Discovering tree structure via forward SDE simulation...")
    unique_labels = sorted(np.unique(y_true))
    
    # Simulate trajectories for each cluster
    print("Simulating cluster trajectories...")
    centroid_trajectories = {}
    cluster_spreads = {}
    
    for label in tqdm(unique_labels, desc="Clusters"):
        current_points = X_data[y_true == label].to(DEVICE)
        
        trajectories = [current_points.mean(axis=0).cpu().numpy()]
        spreads = [torch.mean(torch.norm(
            current_points - current_points.mean(axis=0, keepdim=True), 
            dim=1
        )).item()]
        
        # Simulate forward diffusion
        for t_idx in range(N_STEPS - 1):
            current_points = learned_forward_sde_step(model, current_points, t_idx)
            trajectories.append(current_points.mean(axis=0).cpu().numpy())
            spreads.append(torch.mean(torch.norm(
                current_points - current_points.mean(axis=0, keepdim=True),
                dim=1
            )).item())
        
        centroid_trajectories[label] = np.array(trajectories)
        cluster_spreads[label] = np.array(spreads)
    
    # Agglomerative clustering based on merge times
    print("Building hierarchy from merge times...")
    active_clusters = [frozenset([label]) for label in unique_labels]
    cluster_ids = {frozenset([label]): label for label in unique_labels}
    leaf_counts = {label: 1 for label in unique_labels}
    new_cluster_id_counter = len(unique_labels)
    linkage_info = []
    
    while len(active_clusters) > 1:
        min_merge_time_idx = N_STEPS
        best_pair_to_merge = None
        
        # Find earliest merge
        for c1_set, c2_set in itertools.combinations(active_clusters, 2):
            c1_labels = list(c1_set)
            c2_labels = list(c2_set)
            
            # Average trajectories
            c1_traj = np.mean([centroid_trajectories[l] for l in c1_labels], axis=0)
            c2_traj = np.mean([centroid_trajectories[l] for l in c2_labels], axis=0)
            inter_centroid_dist = np.linalg.norm(c1_traj - c2_traj, axis=1)
            
            # Merge threshold
            c1_spread = np.mean([cluster_spreads[l] for l in c1_labels], axis=0)
            c2_spread = np.mean([cluster_spreads[l] for l in c2_labels], axis=0)
            merge_threshold = c1_spread + c2_spread
            
            # Find first time clusters overlap
            merge_indices = np.where(inter_centroid_dist < merge_threshold)[0]
            
            if len(merge_indices) > 0:
                current_merge_time_idx = merge_indices[0]
                if current_merge_time_idx < min_merge_time_idx:
                    min_merge_time_idx = current_merge_time_idx
                    best_pair_to_merge = (c1_set, c2_set)
        
        # Force merge if no overlap found
        if best_pair_to_merge is None:
            min_dist_at_T = float('inf')
            for c1_set, c2_set in itertools.combinations(active_clusters, 2):
                c1_final = np.mean([centroid_trajectories[l][-1] 
                                   for l in list(c1_set)], axis=0)
                c2_final = np.mean([centroid_trajectories[l][-1] 
                                   for l in list(c2_set)], axis=0)
                dist = np.linalg.norm(c1_final - c2_final)
                if dist < min_dist_at_T:
                    min_dist_at_T = dist
                    best_pair_to_merge = (c1_set, c2_set)
            min_merge_time_idx = N_STEPS - 1
        
        # Perform merge
        c1, c2 = best_pair_to_merge
        id1, id2 = cluster_ids[c1], cluster_ids[c2]
        merge_time_float = (min_merge_time_idx / (N_STEPS - 1)) * T
        
        count1 = leaf_counts.get(id1, 1)
        count2 = leaf_counts.get(id2, 1)
        new_leaf_count = count1 + count2
        
        linkage_info.append([id1, id2, merge_time_float, new_leaf_count])
        
        merged_set = c1.union(c2)
        active_clusters.remove(c1)
        active_clusters.remove(c2)
        active_clusters.append(merged_set)
        
        new_cluster_id = new_cluster_id_counter
        cluster_ids[merged_set] = new_cluster_id
        leaf_counts[new_cluster_id] = new_leaf_count
        new_cluster_id_counter += 1
    
    linkage_matrix = np.array(linkage_info)
    return linkage_matrix

# ============================================================================
# Visualization
# ============================================================================

def plot_results(dataset_name, X_cpu, y_cpu, linkage_matrix):
    """Plot discovered tree structure."""
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(f'Implicit Tree Discovery: {dataset_name.title()}', 
                 fontsize=18)
    
    # Panel (a): Original clusters
    ax = axs[0]
    scatter = ax.scatter(X_cpu[:, 0], X_cpu[:, 1], c=y_cpu, 
                        cmap='viridis', s=10, alpha=0.8)
    ax.set_title('(a) Original Data Clusters', fontsize=14)
    
    # Add cluster labels
    for label in np.unique(y_cpu):
        centroid = X_cpu[y_cpu == label].mean(axis=0)
        ax.text(centroid[0], centroid[1], str(label), 
               fontsize=12, weight='bold', ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.3', fc='white', 
                        alpha=0.7, ec='none'))
    
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Panel (b): Discovered hierarchy
    ax = axs[1]
    dendrogram(linkage_matrix, 
              labels=[str(l) for l in np.unique(y_cpu)], 
              ax=ax)
    ax.set_title('(b) Discovered Hierarchy', fontsize=14)
    ax.set_ylabel("Merge Time (t)", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"implicit_tree_{dataset_name}.png", dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == '__main__':
    dataset_names = ['4_corners', '9_grid', '8_gaussians']
    
    for name in dataset_names:
        print(f"\n{'='*60}")
        print(f"Experiment: {name.replace('_', ' ').title()}")
        print('='*60)
        
        # Generate data
        X, y = get_dataset(name=name, n_samples=3200)
        print(f"Dataset shape: {X.shape}")
        
        # Train diffusion model
        model = train_model(X, nepochs=400)
        
        # Discover tree structure
        linkage_matrix = discover_tree_structure(model, X, y)
        
        # Visualize
        print("\nGenerating visualization...")
        plot_results(name, X.numpy(), y, linkage_matrix)
        
        print(f"✓ Completed {name}")