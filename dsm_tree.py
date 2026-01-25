"""
Experiment 3: Discretized Score Matching for Trees (DSM-Tree)
==============================================================

Trains a neural network to predict tree split decisions at each level,
enabling tree-structured generation through learned dynamics.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ============================================================================
# Configuration
# ============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ============================================================================
# Dataset Loading
# ============================================================================

def get_tabular_datasets():
    """Load standard tabular classification datasets."""
    datasets = {}
    
    # Digits dataset (always available)
    print("Loading Digits dataset...")
    digits = load_digits()
    X_digits, y_digits = digits.data, digits.target
    datasets['Digits (8x8)'] = (X_digits, y_digits)
    
    # Try to load UCI datasets
    try:
        import requests
        import io
        
        # German Credit
        print("Loading German Credit dataset...")
        url = "http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
        column_names = [f"attr_{i}" for i in range(20)] + ["target"]
        df_credit = pd.read_csv(url, sep=" ", header=None, names=column_names)
        df_credit["target"] = df_credit["target"].replace({1: 1, 2: 0})
        X_credit_df = df_credit.drop("target", axis=1)
        X_credit = pd.get_dummies(X_credit_df, drop_first=True).values
        y_credit = df_credit["target"].values
        datasets["German Credit"] = (X_credit, y_credit)
        print("  ✓ German Credit loaded")
    except Exception as e:
        print(f"  ✗ Could not load German Credit: {e}")
    
    return datasets

# ============================================================================
# Phase 1: Ground Truth Tree Generation
# ============================================================================

def train_base_tree(X_train, y_train, max_depth=15):
    """
    Train base tree using Random Forest oracle distillation.
    
    Process:
    1. Train Random Forest oracle (high accuracy)
    2. Get oracle predictions on training data
    3. Distill into single decision tree
    
    Args:
        X_train: Training features
        y_train: Training labels
        max_depth: Maximum tree depth
        
    Returns:
        Fitted DecisionTreeClassifier
    """
    print("  Phase 1: Generating Base Tree via Oracle Distillation...")
    
    # Train oracle
    oracle_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    oracle_model.fit(X_train, y_train)
    
    # Get oracle predictions (soft targets)
    distilled_labels = oracle_model.predict(X_train)
    
    # Distill into single tree
    base_tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    base_tree.fit(X_train, distilled_labels)
    
    print(f"    Base Tree: {base_tree.get_n_leaves()} leaves, "
          f"depth {base_tree.get_depth()}")
    
    return base_tree

# ============================================================================
# Phase 2: Conditional Split Model
# ============================================================================

class ConditionalSplitModel(nn.Module):
    """
    Neural network that predicts split direction at each tree level.
    
    Architecture:
    - Level embedding: Learnable embedding for tree depth
    - MLP: Predicts binary split (left=0, right=1)
    
    Args:
        n_features: Number of input features
        max_depth: Maximum tree depth
        embedding_dim: Dimension of level embeddings
        hidden_dim: Hidden layer dimension
    """
    
    def __init__(self, n_features, max_depth, 
                 embedding_dim=32, hidden_dim=256):
        super().__init__()
        
        # Level embedding for tree depth
        self.level_embedding = nn.Embedding(max_depth, embedding_dim)
        
        # MLP for split prediction
        self.net = nn.Sequential(
            nn.Linear(n_features + embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 2)  # Binary split: left or right
        )
    
    def forward(self, x, j):
        """
        Predict split direction.
        
        Args:
            x: Input features (batch_size, n_features)
            j: Tree level indices (batch_size,)
            
        Returns:
            Logits for left/right split (batch_size, 2)
        """
        j_emb = self.level_embedding(j)
        
        # Handle batch size mismatch
        if j_emb.dim() == 1:
            j_emb = j_emb.unsqueeze(0)
        if j_emb.shape[0] != x.shape[0]:
            j_emb = j_emb.expand(x.shape[0], -1)
        
        return self.net(torch.cat([x, j_emb], dim=1))

# ============================================================================
# DSM-Tree Trainer
# ============================================================================

class DSMTreeTrainer:
    """
    Trainer for Discretized Score Matching on Decision Trees.
    
    Learns to predict split decisions at each level by matching
    the base tree's behavior.
    """
    
    def __init__(self, base_tree, X_train):
        """
        Initialize trainer.
        
        Args:
            base_tree: Fitted DecisionTreeClassifier
            X_train: Training data used to fit base_tree
        """
        self.base_tree = base_tree
        self.X_train_tensor = torch.tensor(
            X_train, dtype=torch.float32, device=DEVICE
        )
        self.max_depth = base_tree.get_depth()
        
        # Compute node depths
        self.node_depths = np.zeros(base_tree.tree_.node_count, dtype=int)
        stack = [(0, 0)]
        while stack:
            node_id, depth = stack.pop()
            self.node_depths[node_id] = depth
            if base_tree.tree_.children_left[node_id] != -1:
                stack.append((base_tree.tree_.children_left[node_id], depth + 1))
                stack.append((base_tree.tree_.children_right[node_id], depth + 1))
        
        # Precompute decision paths
        self.decision_paths = base_tree.decision_path(X_train).toarray()
    
    def get_ground_truth_decision(self, sample_idx, level_j):
        """
        Get ground truth split decision at level j for sample.
        
        Args:
            sample_idx: Index of training sample
            level_j: Tree level to query
            
        Returns:
            0 (left) or 1 (right), or None if sample doesn't reach this level
        """
        path = self.decision_paths[sample_idx]
        nodes_on_path = np.where(path == 1)[0]
        
        # Find node at level j
        node_at_level_j = next(
            (node_id for node_id in nodes_on_path 
             if self.node_depths[node_id] == level_j),
            -1
        )
        
        if node_at_level_j == -1:
            return None  # Sample doesn't reach this level
        
        # Check if it's a leaf
        if self.base_tree.tree_.children_left[node_at_level_j] == -1:
            return None
        
        # Get split decision
        feature = self.base_tree.tree_.feature[node_at_level_j]
        threshold = self.base_tree.tree_.threshold[node_at_level_j]
        
        sample_value = self.X_train_tensor[sample_idx, feature].item()
        
        return 0 if sample_value <= threshold else 1
    
    def train(self, n_steps=30000, batch_size=256, lr=1e-3):
        """
        Train the conditional split model.
        
        Args:
            n_steps: Number of training steps
            batch_size: Batch size
            lr: Learning rate
            
        Returns:
            Trained ConditionalSplitModel
        """
        print("  Phase 2: Training Conditional Split Model...")
        
        n_features = self.X_train_tensor.shape[1]
        model = ConditionalSplitModel(
            n_features, 
            self.max_depth, 
            hidden_dim=256
        ).to(DEVICE)
        
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        
        pbar = tqdm(range(n_steps), desc="Training DSM")
        for step in pbar:
            # Random batch of samples and levels
            sample_indices = np.random.randint(
                0, len(self.X_train_tensor), batch_size
            )
            X_batch = self.X_train_tensor[sample_indices]
            
            # Random levels
            j_batch = torch.randint(
                0, self.max_depth if self.max_depth > 1 else 1,
                (batch_size,),
                device=DEVICE
            )
            
            # Get ground truth decisions
            y_targets = []
            valid_indices = []
            for i, (sample_idx, j) in enumerate(zip(sample_indices, j_batch)):
                target = self.get_ground_truth_decision(sample_idx, j.item())
                if target is not None:
                    y_targets.append(target)
                    valid_indices.append(i)
            
            if not valid_indices:
                continue
            
            # Filter to valid samples
            X_batch_valid = X_batch[valid_indices]
            j_batch_valid = j_batch[valid_indices]
            y_targets_tensor = torch.tensor(
                y_targets, dtype=torch.long, device=DEVICE
            )
            
            # Forward pass
            optimizer.zero_grad()
            y_pred_logits = model(X_batch_valid, j_batch_valid)
            loss = loss_fn(y_pred_logits, y_targets_tensor)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if step % 1000 == 0:
                pbar.set_description(
                    f"Step {step}/{n_steps} | Loss: {loss.item():.4f}"
                )
        
        return model.eval()

# ============================================================================
# Phase 3: Inference
# ============================================================================

@torch.no_grad()
def predict_with_dsm_model(dsm_model, base_tree, X_test):
    """
    Make predictions by traversing tree using DSM model.
    
    Args:
        dsm_model: Trained ConditionalSplitModel
        base_tree: Base decision tree (for structure)
        X_test: Test features
        
    Returns:
        Predicted labels
    """
    print("  Phase 3: Performing Inference...")
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
    y_pred = np.zeros(len(X_test), dtype=int)
    max_depth = base_tree.get_depth()
    
    for i in tqdm(range(len(X_test_tensor)), desc="Predicting"):
        sample = X_test_tensor[i:i+1]
        current_node_id = 0  # Start at root
        
        # Traverse tree
        for j in range(max_depth):
            # Check if leaf
            if base_tree.tree_.children_left[current_node_id] == -1:
                break
            
            # Predict split direction
            j_tensor = torch.tensor([j], device=DEVICE)
            decision_logits = dsm_model(sample, j_tensor)
            decision = torch.argmax(decision_logits, dim=1).item()
            
            # Move to child node
            if decision == 0:
                current_node_id = base_tree.tree_.children_left[current_node_id]
            else:
                current_node_id = base_tree.tree_.children_right[current_node_id]
        
        # Get leaf prediction
        if current_node_id < base_tree.tree_.node_count:
            y_pred[i] = np.argmax(base_tree.tree_.value[current_node_id])
        else:
            y_pred[i] = -1  # Error case
    
    return y_pred

# ============================================================================
# Main Experiment
# ============================================================================

def run_experiment():
    """Run complete DSM-Tree experiment."""
    datasets = get_tabular_datasets()
    all_results_data = []
    
    print(f"\n{'='*60}")
    print("Running DSM-Tree Experiment")
    print('='*60)
    
    for name, (X, y) in datasets.items():
        print(f"\nDataset: {name}")
        print(f"Shape: {X.shape}, Classes: {len(np.unique(y))}")
        
        # Prepare data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42
        )
        
        # Train base tree
        base_tree = train_base_tree(X_train, y_train)
        y_pred_base = base_tree.predict(X_test)
        
        # Train DSM model
        trainer = DSMTreeTrainer(base_tree, X_train)
        dsm_model = trainer.train()
        
        # Inference
        y_pred_dsm = predict_with_dsm_model(dsm_model, base_tree, X_test)
        
        # Evaluate both models
        models = {
            'Base Tree (Baseline)': y_pred_base,
            'DSM-Tree Model': y_pred_dsm
        }
        
        for model_name, y_pred in models.items():
            all_results_data.append({
                "Dataset": name,
                "Model": model_name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Macro F1-Score": f1_score(
                    y_test, y_pred, average='macro', zero_division=0
                ),
                "Cohen's Kappa": cohen_kappa_score(y_test, y_pred)
            })
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results_data)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(results_df.to_string(index=False))
    
    # Plot results
    plot_results(results_df)
    
    return results_df

def plot_results(results_df):
    """Plot comparison of models."""
    plot_df = results_df.pivot(
        index='Dataset', columns='Model', values='Accuracy'
    )[['Base Tree (Baseline)', 'DSM-Tree Model']]
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    plot_df.plot(
        kind='bar', ax=ax, rot=0, width=0.7,
        color={'Base Tree (Baseline)': 'royalblue', 
               'DSM-Tree Model': 'darkorange'}
    )
    
    ax.set_title('DSM-Tree Performance vs. Baseline', fontsize=16, pad=20)
    ax.set_ylabel('Classification Accuracy', fontsize=12)
    ax.set_xlabel('')
    ax.legend(title='Model Type', fontsize=10)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f'{y:.0%}')
    )
    
    # Add value labels
    for container in ax.containers:
        labels = [f'{val*100:.1f}%' for val in container.datavalues]
        ax.bar_label(container, labels=labels, label_type='edge', 
                    padding=5, fontsize=9)
    
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig("dsm_tree_results.png", dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# Run
# ============================================================================

if __name__ == "__main__":
    results = run_experiment()