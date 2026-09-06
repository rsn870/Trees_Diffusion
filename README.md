# Trees to Flows and Back: Unifying Decision Trees and Diffusion Models

> Research experiments exploring connections between decision trees and diffusion models through hierarchical structure, flow matching, and neural tree distillation.

This repository accompanies the paper:

> **Trees to Flows and Back: Unifying Decision Trees and Diffusion Models**
> Sai Niranjan Ramachandran, Suvrit Sra
> *ICML 2026 — 43rd International Conference on Machine Learning*, Seoul, South Korea
> [arXiv:2605.00414](https://arxiv.org/abs/2605.00414)

If you use this code, please cite:

```bibtex
@inproceedings{ramachandran2026trees,
  title     = {Trees to Flows and Back: Unifying Decision Trees and Diffusion Models},
  author    = {Ramachandran, Sai Niranjan and Sra, Suvrit},
  booktitle = {Forty-third International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

## Abstract

Decision trees and diffusion models represent fundamentally different paradigms: one discrete and hierarchical, the other continuous and dynamic. This repository contains four research experiments exploring deep connections between these model classes.

Our experiments demonstrate that:
1. **Diffusion processes implicitly encode hierarchical structure** - clusters merge progressively during forward diffusion, revealing tree-like organization
2. **Trees and diffusion share information-theoretic properties** - both destroy information hierarchically, quantifiable through entropy measures
3. **Tree structure improves flow-based generation** - conditioning flow matching on decision tree paths achieves competitive quality on tabular synthesis with 2× speedup
4. **Neural networks can learn tree decision logic** - DSM-Tree distills tree traversal into differentiable models, matching teacher performance within 2% across benchmarks

Each experiment is self-contained and demonstrates a different facet of the tree-diffusion correspondence. Code is provided for reproduction and adaptation to related problems.

---

## Experiments

### Experiment 1: Implicit Tree Discovery via Diffusion
**What it does:** Discovers hierarchical cluster structure by training a diffusion model and tracking when clusters merge during forward diffusion.

**Key insight:** As noise increases during forward diffusion, nearby clusters merge first. By simulating the forward process and tracking merge times, we can extract an implicit hierarchical tree structure without explicit clustering.

**Run it:**
```bash
python implicit_trees.py
```

**Key variables:**
- `T = 1.0` - Total diffusion time
- `N_STEPS = 100` - Discrete timesteps
- `BETAS` - Noise schedule (how much noise to add at each step)
- `ALPHAS_CUMPROD` - Cumulative signal retention: α̅_t = Π(1-β_i)
- `centroid_trajectories` - Track cluster centers through diffusion
- `cluster_spreads` - Track cluster variance at each timestep

**Output:** Dendrogram showing discovered hierarchy + visualization of clusters at intermediate diffusion times

---

### Experiment 2: Information-Theoretic Comparison (Trees vs Diffusion)
**What it does:** Compares how decision trees and diffusion models progressively lose information, using entropy as a unified metric.

**Key insight:** Both processes destroy information hierarchically:
- **Trees:** Through recursive partitioning (root → leaves)
- **Diffusion:** Through noise addition (t=0 → t=T)

We measure this using:
- **Tree entropy:** Class distribution entropy at each depth
- **Diffusion entropy:** SNR-based measure: 1/(1+SNR) where SNR = α_t/(1-α_t)

**Run it:**
```bash
python entropy_analysis.py
```

**Key variables:**
- `node_depth` - Depth of each node in tree (root=0)
- `entropy_by_depth` - Weighted class entropy at each level
- `SNR = alpha_t / (1 - alpha_t)` - Signal-to-noise ratio at time t
- `entropy_measure = 1 / (1 + SNR)` - Converts SNR to entropy-like metric
- `max_entropy = log2(num_classes)` - Maximum possible entropy for normalization

**Output:** Entropy curves showing parallel information decay + visual prototypes at different tree depths and diffusion times

---

### Experiment 3: DSM-Tree (Discretized Score Matching for Trees)
**What it does:** Trains a neural network to predict tree split decisions at each level, enabling tree-structured generation through learned dynamics.

**Key insight:** Instead of using fixed threshold-based splits, learn a neural network M_θ(x, j) that predicts "left" or "right" at tree level j. This is trained to match a base tree's behavior but can generalize better and integrate into differentiable pipelines.

**Algorithm:**
1. **Phase 1 - Ground Truth:** Train Random Forest oracle → Distill into single decision tree
2. **Phase 2 - Train M_θ(x, j):** For random (sample, level) pairs, predict left/right split using neural net
3. **Phase 3 - Inference:** Traverse tree using neural predictions instead of fixed threshold rules

**Run it:**
```bash
python dsm_tree.py
```

**Key variables:**
- `max_depth` - Maximum tree depth
- `level_embedding` - Learned representation for each tree level j (dimension 32)
- `base_tree` - Ground truth tree trained on oracle predictions
- `decision_paths` - Binary matrix: which nodes each sample visits
- `n_steps = 30000` - Training iterations for neural predictor

**Output:** Performance comparison showing DSM-Tree matches base tree within 2% accuracy across multiple datasets

---

### Experiment 4: Tree-Guided Flow Matching
**What it does:** Combines decision tree structure with flow matching for tabular data synthesis. Trees provide structural conditioning that improves generation quality and computational efficiency.

**Key insight:** Extract tree paths as binary vectors and condition flow model on them. The tree captures hierarchical data structure while flow matching learns smooth transformations. This achieves competitive generation quality with 2× speedup compared to standard diffusion.

**Algorithm:**

**1. Tree Path Encoding:**
```python
# Get binary decision path (sklearn built-in)
decision_path = tree.decision_path(X)  # Which nodes visited?

# Use path directly as encoding
path_encoding = decision_path.toarray()  # (n_samples, n_nodes)
```

**2. Flow Matching with Tree Conditioning:**
```python
# Training
for batch in data:
    t = random_time()  # Sample t ~ Uniform[0,1]
    noise = random_noise()
    
    # Interpolate between data and noise
    x_t = t * x_data + (1-t) * noise
    
    # Get tree path encoding (binary vector)
    path_enc = tree.decision_path(x_data).toarray()
    
    # Predict velocity conditioned on tree
    v_pred = model(x_t, t, path_enc)  # ← Tree conditioning
    
    # Target: data - noise
    v_target = x_data - noise
    
    # Loss
    loss = ||v_pred - v_target||²
```

**3. Sampling:**
```python
# Start from noise
x = random_noise()

# Sample tree path from training data (for target class)
path_enc = tree.decision_path(training_sample).toarray()

# Integrate ODE
for step in range(n_steps):
    t = step / n_steps
    velocity = model(x, t, path_enc)  # Conditioned on tree
    x = x + velocity * dt

return x  # Generated sample following tree structure
```

**Run it:**
```bash
python treeflow_benchmark.py
```

**Key variables:**
- `path_encoding` - Binary tree path representation (n_samples, n_nodes)
- `t` - Flow time parameter [0,1]
- `v = x_data - x_noise` - Velocity field to learn
- `tree.decision_path()` - Sklearn method that returns binary path matrix
- `n_steps = 50` - ODE integration steps 

**Output:** Benchmark results showing competitive quality metrics (Wasserstein distance, TSTR accuracy, correlation error) with 2× computational speedup

---

## Key Concepts

### Path Encoding (Experiment 4)
Converts discrete tree traversal into binary vector:
```
Sample visits: [root, left_child, left_left_leaf]

Binary encoding: [1, 1, 0, 1, 0, 0, ...]
                  ↑  ↑  ↑  ↑  ↑  ↑
               root left skip deep skip skip...
```

Each element is 1 if the sample visited that node, 0 otherwise. This provides structural conditioning for the flow model.

### Flow Matching (Experiment 4)
Learn to transform noise → data:
```
x_0 (noise) --flow--> x_1 (data)
```

By learning velocity field v(x,t) such that:
```
dx/dt = v(x,t)
```

Training: minimize ||v_pred - v_target||² where v_target = x_data - x_noise


### DSM Training (Experiment 3)
For each (sample, level) pair:
```
1. Find which node sample reaches at this level
2. Get ground truth: did it go left (0) or right (1)?
3. Train neural net to predict this binary decision
4. Loss = CrossEntropy(prediction, ground_truth)
```

At test time: traverse tree using neural predictions instead of threshold checks.

---

## Adapting These Ideas

### Idea 1: Extract Tree Structure for Flow Conditioning

```python
# Pseudocode - adapt to your needs

def encode_tree_path(X, tree):
    """Convert tree paths to binary vectors."""
    # Get binary path matrix (sklearn built-in)
    paths = tree.decision_path(X).toarray()  # (n_samples, n_nodes)
    # Each row is 1 where sample visited that node, 0 otherwise
    return paths
```

**Use in your flow model:**
```python
# Standard flow model
def forward(self, x, t):
    return self.net(concat([x, t]))

# With tree conditioning (requires architecture modification)
def forward(self, x, t, path_encoding):
    # Concatenate tree structure
    h = concat([x, t, path_encoding])  # ← Add binary tree path
    return self.net(h)  # ← Architecture must handle larger input dimension
```

### Idea 2: Learn Neural Network to Mimic Tree Splits (DSM-Tree)

```python
# Pseudocode - can be integrated into your classification pipeline

class NeuralTreePredictor:
    def __init__(self, n_features, max_depth):
        self.level_embedding = Embedding(max_depth, 32)
        self.mlp = MLP(n_features + 32 → 2)  # Binary: left or right
    
    def forward(self, x, level):
        level_emb = self.level_embedding(level)
        return self.mlp(concat([x, level_emb]))

# Training (learns to match a base decision tree)
for (sample, level) in random_pairs:
    # Get ground truth from base tree
    ground_truth_split = get_split_from_tree(sample, level, base_tree)
    
    # Predict
    logits = model(sample, level)
    
    # Train
    loss = CrossEntropy(logits, ground_truth_split)

# Prediction (traverse tree using neural net predictions)
def predict(x):
    node = root
    for level in range(max_depth):
        if is_leaf(node): break
        
        # Neural net predicts left (0) or right (1)
        split_direction = argmax(model(x, level))
        
        # Move to child node
        node = left_child(node) if split_direction == 0 else right_child(node)
    
    return class_at_leaf(node)
```

**Why use this instead of regular decision tree?**
- Can learn smoother decision boundaries
- May generalize better than hard threshold splits  
- Integrates into differentiable pipelines
- Typically matches base tree performance within 2%

### Idea 3: Discover Hierarchical Structure from Diffusion

```python
# Pseudocode

# 1. Train diffusion model
model = train_diffusion(data)

# 2. Simulate forward diffusion for each cluster
for cluster in clusters:
    trajectories[cluster] = []
    spreads[cluster] = []
    
    points = data[cluster]
    for t in timesteps:
        points = learnt_forward_diffusion_step(points, t, model)  # reverse learnt forward PF-ODE
        trajectories[cluster].append(mean(points))
        spreads[cluster].append(std(points))

# 3. Find merge times
for (cluster_i, cluster_j) in all_pairs:
    distance = ||trajectory_i - trajectory_j||
    threshold = spread_i + spread_j
    
    # When do they merge?
    merge_time = first_time(distance < threshold)

# 4. Build tree from merge times (agglomerative clustering)
linkage_matrix = build_hierarchy_from_merge_times(merge_times)
```

---

## Requirements

```
torch>=2.0.0
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
tqdm>=4.62.0
torchvision>=0.15.0  # For Experiment 2
```

Install:
```bash
pip install torch numpy scipy scikit-learn pandas matplotlib seaborn tqdm torchvision
```

---

## Repository Structure

```
.
├── README.md
├── implicit_trees.py       # Hierarchical discovery via diffusion
├── entropy_analysis.py     # Information-theoretic comparison
├── dsm_tree.py             # Neural tree distillation
└── treeflow_benchmark.py   # Tree-guided flow matching
```

---

## Key Results

| Experiment | Key Finding |
|------------|-------------|
| **Experiment 1** | Diffusion processes reveal implicit cluster hierarchies through progressive merging |
| **Experiment 2** | Trees and diffusion exhibit parallel entropy decay patterns despite different mechanisms |
| **Experiment 3** | DSM-Tree matches teacher tree performance within 2% across benchmarks |
| **Experiment 4** | Tree-conditioned flow achieves competitive quality with 2× computational speedup |

---

## Notes

- These are **research experiments**, not production libraries
- Code is meant to be **read, understood, and adapted** to your specific use case
- Each experiment is **self-contained** and can be run independently
- Focus is on **demonstrating conceptual connections** between trees and diffusion
- Pseudocode sections show how to adapt core ideas to other problems

---

## Citation

```bibtex
@inproceedings{ramachandran2026trees,
  title     = {Trees to Flows and Back: Unifying Decision Trees and Diffusion Models},
  author    = {Ramachandran, Sai Niranjan and Sra, Suvrit},
  booktitle = {Forty-third International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

## License

MIT
