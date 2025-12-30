# Layer-wise Harmfulness and Refusal Analysis

This directory contains code for analyzing where harmfulness and refusal concepts "crystallize" in transformer models, extending the harmfulness/refusal probe paper.

## Overview

### Research Questions

1. **Layer-wise Dynamics** (Priority #1)
   - Where do harmfulness and refusal representations crystallize?
   - Are there specific "critical layers" where these concepts form?
   - Do early layers encode semantic harmfulness while late layers handle refusal?

2. **Category-Specific Representations** (Priority #3)
   - Is harmfulness category-specific or universal?
   - Do different harm categories (violence, illegal, hate speech) have distinct subspaces?
   - Does refusal have a universal direction while harmfulness is category-specific?

3. **Compositional Harmfulness** (Priority #2)
   - How does context modulate harmfulness representations?
   - Does educational framing reduce harmfulness signal?
   - Can we decompose harmfulness into base + context components?

## Files

### Core Implementation

- **`layerwise_analysis.py`**: Main analysis module
  - Probe training for all layers
  - Direction extraction and comparison
  - Cluster quality metrics (silhouette scores)
  - Category-specific analysis
  - Compositional harmfulness experiments
  - Visualization utilities

### Notebooks

- **`Harmfulness_and_Refusal_Probes.ipynb`**: Base implementation
  - Model wrapper (ChatModel)
  - Dataset loading (AdvBench, XSTest, HH-RLHF)
  - Activation extraction
  - Refusal detection
  - Data compression and caching

- **`Layerwise_Harmfulness_Refusal_Analysis.ipynb`**: Main experiments
  - Comprehensive layer-wise analysis
  - Critical layer identification
  - Category specificity experiments
  - Position comparison (t_inst vs t_postinst)
  - Visualization and interpretation

## Quick Start

### 1. Setup

```bash
# Install dependencies
pip install torch transformers numpy pandas matplotlib seaborn scikit-learn tqdm datasets
```

### 2. Collect Activations

First, run the base notebook to collect activations:

```python
# In Harmfulness_and_Refusal_Probes.ipynb

# 1. Download datasets
download_all_datasets()

# 2. Load model
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
chat_model = ChatModel(MODEL_NAME, LLAMA3_TEMPLATE)

# 3. Load datasets
loader = DatasetLoader()
examples = loader.load_mixed_dataset(
    n_harmful=250,
    n_harmless=250,
    harmless_source="xstest"
)

# 4. Process and cache
compressed = process_dataset_batch(
    chat_model,
    examples,
    cache_path="cache/llama3_1b_mixed_500.pkl",
    batch_size=50
)
```

### 3. Run Layer-wise Analysis

Then run the layer-wise analysis notebook:

```python
# In Layerwise_Harmfulness_Refusal_Analysis.ipynb

# 1. Load cached activations
with open("cache/llama3_1b_mixed_500.pkl", 'rb') as f:
    compressed_data = pickle.load(f)

# 2. Run comprehensive analysis
metrics = analyze_layerwise_dynamics(
    compressed_data,
    position="inst",
    test_size=0.3,
    C=1.0
)

# 3. Visualize
plot_layerwise_metrics(metrics, save_path="figures/layerwise_dynamics.png")

# 4. Category analysis
best_layer = np.argmax(metrics.harmfulness_acc)
cat_analysis = analyze_category_specificity(
    compressed_data,
    layer=best_layer,
    position="inst"
)
```

## Key Functions

### Training Probes

```python
from layerwise_analysis import train_layerwise_probes

# Train probes for all layers
probes = train_layerwise_probes(
    compressed_data,
    position="inst",  # or "postinst"
    concept="harmfulness",  # or "refusal"
    test_size=0.3,
    C=1.0
)
```

### Comprehensive Analysis

```python
from layerwise_analysis import analyze_layerwise_dynamics

# Run full analysis (trains both harmfulness and refusal probes)
metrics = analyze_layerwise_dynamics(
    compressed_data,
    position="inst",
    test_size=0.3
)

# Access results
print(f"Best harmfulness layer: {np.argmax(metrics.harmfulness_acc)}")
print(f"Best refusal layer: {np.argmax(metrics.refusal_acc)}")
```

### Category Analysis

```python
from layerwise_analysis import analyze_category_specificity

# Analyze category-specific representations
cat_analysis = analyze_category_specificity(
    compressed_data,
    layer=15,  # Choose based on probe performance
    position="inst",
    test_size=0.3
)

# Check cross-category generalization
print(cat_analysis.generalization_matrix)
print(cat_analysis.similarity_matrix)
```

### Visualization

```python
from layerwise_analysis import plot_layerwise_metrics, plot_category_analysis

# Visualize layer-wise dynamics
plot_layerwise_metrics(metrics, save_path="figures/layerwise_dynamics.png")

# Visualize category analysis
plot_category_analysis(cat_analysis, save_path="figures/category_analysis.png")
```

## Expected Outputs

### Layer-wise Metrics

- **Probe accuracy by layer**: Which layers have strongest harmfulness/refusal signal
- **AUC scores**: Discrimination quality
- **Silhouette scores**: Cluster separation quality
- **Direction similarity**: How directions change across layers
- **Critical layers**: Where concepts crystallize

### Category Analysis

- **Category directions**: Probe weights for each harm category
- **Similarity matrix**: Cosine similarity between category directions
- **Generalization matrix**: Cross-category test accuracies
- **Interpretation**: Whether harmfulness is universal or category-specific

## Interpreting Results

### Layer-wise Dynamics

**Good signals**:
- Probe accuracy > 70% indicates strong representation
- Silhouette score > 0.3 indicates good cluster separation
- Direction similarity > 0.9 indicates stable representation

**Critical layer identification**:
- First layer where accuracy exceeds threshold (e.g., 70%)
- This is where the concept "crystallizes"

**Hypotheses to test**:
- ✅ Harmfulness appears in early layers (semantic encoding)
- ✅ Refusal appears in late layers (decision making)
- ✅ Directions stabilize after crystallization point

### Category Specificity

**Universal representation**:
- High similarity (>0.8) between all category pairs
- High cross-generalization (>0.7)
- Interpretation: Single universal "harmfulness" concept

**Category-specific representation**:
- Low off-diagonal similarity (<0.5)
- Poor cross-generalization (<0.6)
- Interpretation: Distinct subspaces for each harm type

**Mixed pattern**:
- Some categories cluster together (similar domains)
- Others are distinct (different harm types)
- Interpretation: Hierarchical structure

## Advanced Experiments

### Temporal Evolution

Track how representations change during generation:

```python
# This requires modifying the forward pass to extract
# activations at multiple generation steps
# See layerwise_analysis.py for implementation outline
```

### Steering by Layer

Test intervention effectiveness:

```python
# Extract best direction from layer analysis
best_layer = np.argmax(metrics.harmfulness_acc)
direction = metrics.harmfulness_directions[best_layer]

# Apply steering at different layers and measure effect
# (Implementation depends on your steering method)
```

### Cross-Model Comparison

Analyze whether patterns transfer:

```python
# Run analysis on multiple model sizes
models = ["Llama-3.2-1B", "Llama-3.2-3B"]
# Compare critical layers, direction similarity, etc.
```

## Troubleshooting

### Low probe accuracy (<60%)

- Check label balance (should be roughly 50/50)
- Try different regularization (C parameter)
- Verify activations are extracted correctly
- Try different position (inst vs postinst)

### Category analysis fails

- Need at least 50+ harmful examples
- Try manual categorization instead of clustering
- Reduce number of categories

### Memory issues

- Process data in smaller batches
- Use float16 instead of float32
- Clear cache between operations: `torch.cuda.empty_cache()`

## Citation

If you use this code, please cite the original harmfulness/refusal paper and acknowledge this extension work.

## Contact

For questions or issues, please open an issue on the GitHub repository.
