"""
Experiment 2: Information-Theoretic Comparison of Trees and Diffusion
======================================================================

Compares entropy-based information decay in decision trees versus
diffusion models using a unified metric.
"""

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import entropy as scipy_entropy
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# Configuration
# ============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Diffusion parameters
T = 1.0
N_STEPS = 100
BETAS = torch.linspace(0.0001, 0.02, N_STEPS)
ALPHAS_CUMPROD = torch.cumprod(1. - BETAS, axis=0)

# ============================================================================
# Dataset Loading
# ============================================================================

def get_dataset(name='mnist', train_samples=10000, test_samples=2000):
    """Load and prepare image classification datasets."""
    
    if name == 'mnist':
        dataset_class = torchvision.datasets.MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    elif name == 'fmnist':
        dataset_class = torchvision.datasets.FashionMNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    elif name == 'usps':
        dataset_class = torchvision.datasets.USPS
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((28, 28)),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    # Load datasets
    train_set = dataset_class(root='./data', train=True, 
                             download=True, transform=transform)
    test_set = dataset_class(root='./data', train=False, 
                            download=True, transform=transform)
    
    # Adjust sample sizes
    actual_train_size = min(train_samples, len(train_set))
    actual_test_size = min(test_samples, len(test_set))
    
    # Create loaders
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=actual_train_size
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=actual_test_size
    )
    
    # Extract data
    X_train, y_train = next(iter(train_loader))
    X_test, y_test = next(iter(test_loader))
    
    # Flatten for tree
    X_train_flat = X_train.view(actual_train_size, -1).numpy()
    X_test_flat = X_test.view(actual_test_size, -1).numpy()
    y_train_np = y_train.numpy()
    y_test_np = y_test.numpy()
    
    return (
        (X_train_flat, y_train_np, X_test_flat, y_test_np),
        (X_train, y_train_np)
    )

# ============================================================================
# Decision Tree Entropy Analysis
# ============================================================================

def train_and_analyze_tree_entropy(X_train, y_train, X_test, y_test):
    """
    Train decision tree and compute entropy at each depth level.
    
    Returns:
        normalized_depths: Progress from leaf (0) to root (1)
        entropies: Normalized entropy at each depth
        dt: Trained decision tree
    """
    print("  Training Decision Tree...")
    dt = DecisionTreeClassifier(max_depth=15, random_state=42)
    dt.fit(X_train, y_train)
    
    y_pred = dt.predict(X_test)
    print(f"  Decision Tree Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    tree = dt.tree_
    max_depth = tree.max_depth
    
    # Calculate node depths
    node_depth = np.zeros(shape=tree.node_count, dtype=np.int64)
    stack = [(0, 0)]  # (node_id, depth)
    
    while len(stack) > 0:
        node_id, depth = stack.pop()
        node_depth[node_id] = depth
        
        if tree.children_left[node_id] != tree.children_right[node_id]:
            stack.append((tree.children_left[node_id], depth + 1))
            stack.append((tree.children_right[node_id], depth + 1))
    
    # Calculate entropy by depth (weighted by samples)
    entropy_by_depth = {}
    num_classes = len(np.unique(y_train))
    max_entropy = np.log2(num_classes)  # Maximum entropy
    
    for depth in range(max_depth + 1):
        nodes_at_depth = np.where(node_depth == depth)[0]
        total_samples_at_depth = 0
        weighted_entropy = 0
        
        for node_id in nodes_at_depth:
            class_counts = tree.value[node_id][0]
            node_samples = tree.n_node_samples[node_id]
            
            if node_samples > 0:
                # Class distribution
                class_probs = class_counts / node_samples
                # Entropy in bits
                node_entropy = scipy_entropy(class_probs, base=2)
                
                weighted_entropy += node_entropy * node_samples
                total_samples_at_depth += node_samples
        
        if total_samples_at_depth > 0:
            # Normalized entropy
            entropy_by_depth[depth] = (
                weighted_entropy / total_samples_at_depth / max_entropy
            )
    
    depths = np.array(list(entropy_by_depth.keys()))
    entropies = np.array(list(entropy_by_depth.values()))
    normalized_depths = depths / max_depth
    
    # Return in order: leaf (depth=max) -> root (depth=0)
    # So progress goes from 0 to 1
    return (1 - normalized_depths)[::-1], entropies[::-1], dt

# ============================================================================
# Diffusion Entropy Analysis
# ============================================================================

def calculate_diffusion_entropy_from_snr():
    """
    Calculate entropy measure from SNR during forward diffusion.
    
    SNR = alpha_t / (1 - alpha_t) measures signal-to-noise ratio.
    We convert to entropy: normalized_entropy = 1 / (1 + SNR)
    
    This maps:
    - High SNR (clean) → Low entropy (low uncertainty)
    - Low SNR (noisy) → High entropy (high uncertainty)
    
    Returns:
        normalized_time: Progress from t=0 to t=T
        entropy_measure: Entropy-like metric from SNR
    """
    print("  Calculating entropy from SNR...")
    
    t_steps = torch.arange(0, N_STEPS)
    alphas_bar = ALPHAS_CUMPROD[t_steps]
    
    # SNR = signal variance / noise variance
    snr = alphas_bar / (1 - alphas_bar + 1e-9)
    
    # Convert SNR to entropy-like measure
    entropy_measure = 1.0 / (1.0 + snr.numpy())
    
    normalized_time = t_steps.numpy() / (N_STEPS - 1)
    
    return normalized_time, entropy_measure

def forward_diffusion(x0, t_idx):
    """Apply forward diffusion at timestep t_idx."""
    noise = torch.randn_like(x0)
    t_idx = torch.tensor(t_idx).long()
    
    sqrt_alpha_t = torch.sqrt(ALPHAS_CUMPROD[t_idx]).view(-1, 1, 1)
    sqrt_one_minus_alpha_t = torch.sqrt(1. - ALPHAS_CUMPROD[t_idx]).view(-1, 1, 1)
    
    return sqrt_alpha_t * x0 + sqrt_one_minus_alpha_t * noise

# ============================================================================
# Visualization
# ============================================================================

def create_and_save_combined_plot(name, dt, X_train_flat, X_train_img, y_train,
                                  dt_norm_depths, dt_entropy, 
                                  diff_norm_time, diff_entropy):
    """Create comprehensive visualization comparing tree and diffusion."""
    print(f"  Generating visualization for {name.upper()}...")
    tree = dt.tree_
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 2], hspace=0.4)
    fig.suptitle(
        f'Information-Theoretic Analysis: Decision Tree vs. Diffusion\n'
        f'Dataset: {name.upper()}',
        fontsize=20, y=0.98
    )
    
    # ========================================================================
    # Top Plot: Entropy Comparison
    # ========================================================================
    ax_main = fig.add_subplot(gs[0])
    
    ax_main.plot(dt_norm_depths, dt_entropy, 'o-', 
                label='Decision Tree (Class Distribution Entropy)',
                color='blue', linewidth=2, markersize=8)
    ax_main.plot(diff_norm_time, diff_entropy, 's-', 
                label='Diffusion (SNR-based Entropy: 1/(1+SNR))',
                color='red', markersize=6, linewidth=2)
    
    ax_main.set_xlabel("Process Progression (Leaf/t=0 → Root/t=T)", 
                      fontsize=14)
    ax_main.set_ylabel("Normalized Entropy (Information Uncertainty)", 
                      fontsize=14)
    ax_main.set_title("Entropy-Based Information Decay", fontsize=16, pad=10)
    ax_main.set_ylim(-0.05, 1.05)
    ax_main.legend(fontsize=11, loc='upper left')
    ax_main.grid(True, linestyle='--', alpha=0.6)
    
    # ========================================================================
    # Bottom Plot: Visual Prototypes
    # ========================================================================
    ax_prototypes_container = fig.add_subplot(gs[1])
    ax_prototypes_container.axis('off')
    ax_prototypes_container.set_title(
        "Visual Progression of Information Loss",
        fontsize=16, pad=20
    )
    gs_prototypes = gs[1].subgridspec(2, 6, wspace=0, hspace=0)
    
    # Row labels
    ax_tree_label = fig.add_subplot(gs_prototypes[0, 0])
    ax_tree_label.text(0.5, 0.5, 'Decision Tree\nPrototypes', 
                      ha='center', va='center',
                      fontsize=12, weight='bold')
    ax_tree_label.axis('off')
    
    ax_diff_label = fig.add_subplot(gs_prototypes[1, 0])
    ax_diff_label.text(0.5, 0.5, 'Diffusion\nPrototypes', 
                      ha='center', va='center',
                      fontsize=12, weight='bold')
    ax_diff_label.axis('off')
    
    # Choose example class
    if name == 'fmnist':
        example_class = 8  # Bag
    elif name == 'usps':
        example_class = 5  # Digit 5
    else:
        example_class = 5  # Digit 5 for MNIST
    
    example_idx = np.where(y_train == example_class)[0][0]
    example_image_flat = X_train_flat[example_idx:example_idx+1]
    example_image_tensor = X_train_img[example_idx:example_idx+1]
    
    # ========================================================================
    # Tree Prototypes: Leaf -> Root
    # ========================================================================
    path = dt.decision_path(example_image_flat).toarray()[0]
    path_nodes = np.where(path == 1)[0]
    
    # Compute node depths
    node_depth = np.zeros(shape=tree.node_count, dtype=np.int64)
    stack = [(0, 0)]
    while len(stack) > 0:
        node_id, depth = stack.pop()
        node_depth[node_id] = depth
        if tree.children_left[node_id] != tree.children_right[node_id]:
            stack.append((tree.children_left[node_id], depth+1))
            stack.append((tree.children_right[node_id], depth+1))
    
    path_depths = node_depth[path_nodes]
    unique_path_depths = sorted(np.unique(path_depths))
    depths_to_show_indices = np.linspace(0, len(unique_path_depths)-1, 5, dtype=int)
    nodes_to_show = [
        path_nodes[path_depths == unique_path_depths[i]][0] 
        for i in depths_to_show_indices
    ]
    
    decision_path_all = dt.decision_path(X_train_flat)
    
    for i, node_id in enumerate(reversed(nodes_to_show)):
        ax_proto = fig.add_subplot(gs_prototypes[0, i+1])
        samples_in_node = np.where(
            decision_path_all.toarray()[:, node_id] == 1
        )[0]
        prototype = X_train_flat[samples_in_node].mean(axis=0)
        
        # Add noise proportional to entropy
        class_counts = tree.value[node_id][0]
        node_samples = tree.n_node_samples[node_id]
        class_probs = class_counts / node_samples
        node_entropy = scipy_entropy(class_probs, base=2) if node_samples > 0 else np.log2(len(np.unique(y_train)))
        max_entropy = np.log2(len(np.unique(y_train)))
        noise_level = 0.4 * (node_entropy / max_entropy)
        
        if i == 0:
            noise_level = 0.0
        
        noisy_prototype = prototype + np.random.randn(*prototype.shape) * noise_level
        ax_proto.imshow(noisy_prototype.reshape(28, 28), cmap='gray', 
                       interpolation='bicubic')
        ax_proto.axis('off')
    
    # ========================================================================
    # Diffusion Prototypes (t=0 -> t=T)
    # ========================================================================
    time_points_for_proto = np.linspace(0, N_STEPS-1, 5, dtype=int)
    for i, t_int in enumerate(time_points_for_proto):
        ax_proto = fig.add_subplot(gs_prototypes[1, i+1])
        noised_image = forward_diffusion(example_image_tensor, t_int)
        ax_proto.imshow(noised_image.numpy().reshape(28, 28), cmap='gray', 
                       interpolation='bicubic')
        ax_proto.axis('off')
    
    # Add arrow
    ax_prototypes_container.annotate(
        '', xy=(0.95, -0.1), xytext=(0.15, -0.1),
        arrowprops=dict(facecolor='red', shrink=0.05, width=3, headwidth=10),
        xycoords='axes fraction'
    )
    
    plt.savefig(f"combined_analysis_{name}.png", dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == '__main__':
    datasets = ['mnist', 'fmnist', 'usps']
    
    for name in datasets:
        print(f"\n{'='*60}")
        print(f"Experiment: {name.upper()}")
        print('='*60)
        
        # Load data
        (X_train_flat, y_train_flat, X_test_flat, y_test_np), \
        (X_train_img, y_train_img) = get_dataset(name)
        
        # Decision tree entropy analysis
        dt_norm_depths, dt_entropy, trained_dt = train_and_analyze_tree_entropy(
            X_train_flat, y_train_flat, X_test_flat, y_test_np
        )
        
        # Diffusion entropy from SNR
        diff_norm_time, diff_entropy = calculate_diffusion_entropy_from_snr()
        
        # Create visualization
        create_and_save_combined_plot(
            name, trained_dt, X_train_flat, X_train_img, y_train_img,
            dt_norm_depths, dt_entropy, diff_norm_time, diff_entropy
        )
        
        print(f"✓ Completed {name.upper()}")