"""
Experiment 4: TreeFlow - Tree-Guided Flow Matching for Tabular Data
====================================================================

Benchmarks TreeFlow against standard tabular synthesis baselines.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import time
import warnings
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ============================================================================
# Configuration
# ============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ============================================================================
# Dataset Loading
# ============================================================================

def load_benchmark_suite():
    """Load standard sklearn datasets for benchmarking."""
    datasets = {}
    
    print("Loading datasets...")
    
    # Breast Cancer
    datasets['Cancer'] = load_breast_cancer(return_X_y=True)
    
    # Diabetes (convert to binary classification)
    X_d, y_d = load_diabetes(return_X_y=True)
    datasets['Diabetes'] = (X_d, (y_d > 150).astype(int))
    
    # Wine (binary: class 0 vs rest)
    X_w, y_w = load_wine(return_X_y=True)
    datasets['Wine'] = (X_w, (y_w == 0).astype(int))
    
    return datasets

# ============================================================================
# TreeFlow Components
# ============================================================================

class TreePathEncoder:
    """Encodes decision tree paths as continuous vectors."""
    
    def __init__(self, tree):
        self.tree = tree
        self.node_count = tree.tree_.node_count
        self.depths = self._compute_depths()
    
    def _compute_depths(self):
        """Compute depth of each node."""
        depths = np.zeros(self.node_count)
        stack = [(0, 0)]
        while stack:
            idx, d = stack.pop()
            depths[idx] = d
            if self.tree.tree_.children_left[idx] != -1:
                stack.append((self.tree.tree_.children_left[idx], d+1))
                stack.append((self.tree.tree_.children_right[idx], d+1))
        return depths
    
    def encode(self, X):
        """Encode samples as depth-weighted path vectors."""
        decision_path = self.tree.decision_path(X).toarray()
        depth_weights = 1.0 / (self.depths + 1.0)
        return decision_path * depth_weights

class TreeFlowModel(nn.Module):
    """Neural network for tree-guided flow matching."""
    
    def __init__(self, in_dim, p_dim, num_classes):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(in_dim + p_dim + 1 + 16, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, in_dim)
        )
        
        # Class embedding
        self.y_emb = nn.Embedding(num_classes, 16)
    
    def forward(self, x, t, p, y):
        """
        Predict velocity field.
        
        Args:
            x: Current state (batch, features)
            t: Time values (batch, 1) in [0,1]
            p: Path encodings (batch, nodes)
            y: Class labels (batch,)
        """
        return self.net(torch.cat([x, t, p, self.y_emb(y)], dim=-1))

# ============================================================================
# Training
# ============================================================================

def train_treeflow(X_train, y_train, max_depth=10, epochs=1000, lr=1e-3):
    """
    Train TreeFlow model.
    
    Process:
    1. Train decision tree
    2. Encode tree paths
    3. Train flow matching model
    
    Returns:
        model, encoder, runtime
    """
    start = time.time()
    
    # Train tree
    dt = DecisionTreeClassifier(max_depth=max_depth).fit(X_train, y_train)
    enc = TreePathEncoder(dt)
    
    # Prepare data
    num_classes = int(np.max(y_train) + 1)
    tf = TreeFlowModel(
        X_train.shape[1], enc.node_count, num_classes
    ).to(DEVICE)
    opt = optim.AdamW(tf.parameters(), lr=lr)
    
    X_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    Y_t = torch.tensor(y_train, dtype=torch.long).to(DEVICE)
    P_t = torch.tensor(enc.encode(X_train), dtype=torch.float32).to(DEVICE)
    
    # Training loop
    tf.train()
    for _ in tqdm(range(epochs), desc="Training TreeFlow"):
        # Random time
        t_step = torch.rand(len(X_t), 1).to(DEVICE)
        
        # Random noise
        x0 = torch.randn_like(X_t)
        
        # Conditional Flow Matching
        xt = t_step * X_t + (1 - t_step) * x0
        v_target = X_t - x0
        v_pred = tf(xt, t_step, P_t, Y_t)
        
        # Update
        loss = nn.functional.mse_loss(v_pred, v_target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    
    runtime = time.time() - start
    
    return tf.eval(), enc, runtime

# ============================================================================
# Sampling
# ============================================================================

@torch.no_grad()
def sample_treeflow(model, encoder, y_target, X_train, y_train, n_steps=50):
    """
    Generate samples using TreeFlow.
    
    Args:
        model: Trained TreeFlowModel
        encoder: TreePathEncoder
        y_target: Target labels for generation
        X_train: Training data (for path sampling)
        y_train: Training labels
        n_steps: Integration steps
    
    Returns:
        Generated samples
    """
    # Sample path encodings from training data with matching labels
    p_enc_tr = encoder.encode(X_train)
    p_idx = [
        np.random.choice(np.where(y_train == v)[0]) 
        for v in y_target
    ]
    p_gen = torch.tensor(
        p_enc_tr[p_idx], dtype=torch.float32
    ).to(DEVICE)
    
    # Initialize from noise
    x_gen = torch.randn(len(y_target), X_train.shape[1]).to(DEVICE)
    y_gen = torch.tensor(y_target, dtype=torch.long).to(DEVICE)
    
    # Integrate ODE
    for i in range(n_steps):
        t_val = torch.full((len(y_target), 1), i/n_steps).to(DEVICE)
        x_gen += model(x_gen, t_val, p_gen, y_gen) * (1.0/n_steps)
    
    return x_gen.cpu().numpy()

# ============================================================================
# Evaluation Metrics
# ============================================================================

def compute_correlation_error(real, fake):
    """Frobenius norm of correlation matrix difference."""
    corr_real = np.nan_to_num(np.corrcoef(real, rowvar=False))
    corr_fake = np.nan_to_num(np.corrcoef(fake, rowvar=False))
    return np.linalg.norm(corr_real - corr_fake, ord='fro')

# ============================================================================
# Benchmark
# ============================================================================

def run_benchmark(n_runs=3):
    """Run complete benchmark comparing TreeFlow to baselines."""
    suite = load_benchmark_suite()
    results = []
    
    for name, (X, y) in suite.items():
        print(f"\n{'='*60}")
        print(f"Benchmarking: {name}")
        print('='*60)
        
        num_classes = int(np.max(y) + 1)
        
        for run in range(n_runs):
            print(f"\nRun {run + 1}/{n_runs}")
            
            # Split data
            X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
                X, y, test_size=0.3, random_state=run
            )
            
            # Standardize
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr_raw)
            X_te_s = scaler.transform(X_te_raw)
            
            # Train TreeFlow
            model, encoder, runtime = train_treeflow(
                X_tr_s, y_tr, epochs=800
            )
            
            # Generate samples
            X_fake = sample_treeflow(
                model, encoder, y_te, X_tr_s, y_tr
            )
            
            # Evaluate
            # 1. Wasserstein distance (fidelity)
            wd = np.mean([
                wasserstein_distance(X_te_s[:,i], X_fake[:,i]) 
                for i in range(X.shape[1])
            ])
            
            # 2. TSTR accuracy (utility)
            clf = RandomForestClassifier(n_estimators=50, random_state=42)
            clf.fit(X_fake, y_te)
            acc = accuracy_score(y_te, clf.predict(X_te_s))
            
            # 3. Correlation error (structure)
            corr_err = compute_correlation_error(X_te_s, X_fake)
            
            results.append({
                "Dataset": name,
                "Run": run,
                "Wasserstein": wd,
                "TSTR_Acc": acc,
                "Corr_Error": corr_err,
                "Runtime": runtime
            })
    
    return pd.DataFrame(results)

# ============================================================================
# Visualization
# ============================================================================

def plot_results(df):
    """Plot benchmark results."""
    sns.set(style="whitegrid", font_scale=1.1)
    
    metrics = [
        ('TSTR_Acc', 'Utility (TSTR Accuracy ↑)'),
        ('Wasserstein', 'Fidelity (Wasserstein ↓)'),
        ('Corr_Error', 'Structure (Correlation Error ↓)'),
        ('Runtime', 'Efficiency (Training Time ↓)')
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, (col, title) in enumerate(metrics):
        sns.barplot(
            data=df, x="Dataset", y=col, ax=axes[i],
            color='#2ecc71', capsize=.1, errorbar='sd'
        )
        axes[i].set_title(title, fontweight='bold', fontsize=14)
        axes[i].set_ylabel("")
        axes[i].set_xlabel("")
    
    plt.tight_layout()
    plt.savefig("treeflow_benchmark.png", dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # Run benchmark
    results_df = run_benchmark(n_runs=3)
    
    # Print results
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    
    stats = results_df.groupby('Dataset').agg(['mean', 'std']).round(4)
    print(stats)
    
    # Plot
    plot_results(results_df)
    
    print("\n✓ Benchmark complete!")