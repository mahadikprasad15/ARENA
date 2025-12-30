"""
Layer-wise Analysis of Harmfulness and Refusal Representations

This module provides tools for analyzing where harmfulness and refusal concepts
"crystallize" in transformer models, building on the harmfulness/refusal probe paper.

Key capabilities:
1. Layer-wise probe training and direction extraction
2. Critical layer identification
3. Category-specific harmfulness analysis
4. Temporal evolution tracking
5. Compositional harmfulness experiments
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import pickle
from pathlib import Path
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ProbeResult:
    """Results from training a single probe."""
    layer: int
    position: str  # "inst" or "postinst"
    concept: str   # "harmfulness" or "refusal"
    accuracy: float
    auc: float
    direction: np.ndarray  # The probe direction vector
    train_acc: float
    test_acc: float
    n_samples: int


@dataclass
class LayerWiseMetrics:
    """Comprehensive metrics for all layers."""
    layers: List[int]

    # Probe performance
    harmfulness_acc: np.ndarray  # [n_layers]
    refusal_acc: np.ndarray
    harmfulness_auc: np.ndarray
    refusal_auc: np.ndarray

    # Cluster quality
    harmfulness_silhouette: np.ndarray
    refusal_silhouette: np.ndarray

    # Directions
    harmfulness_directions: List[np.ndarray]  # List of [d_model] vectors
    refusal_directions: List[np.ndarray]

    # Direction similarity across layers
    harmfulness_direction_similarity: np.ndarray  # [n_layers, n_layers]
    refusal_direction_similarity: np.ndarray


@dataclass
class CategoryAnalysis:
    """Analysis of category-specific harmfulness representations."""
    categories: List[str]

    # Per-category directions
    category_directions: Dict[str, np.ndarray]

    # Cross-category similarity matrix
    similarity_matrix: np.ndarray  # [n_categories, n_categories]

    # Cross-category generalization
    generalization_matrix: np.ndarray  # [n_categories, n_categories] - test accuracy

    # Layer-wise category specificity
    layer_specificity: np.ndarray  # [n_layers, n_categories]


# ============================================================================
# Probe Training
# ============================================================================

def extract_activations_by_label(
    compressed_data: List,
    position: str = "inst",
    label_type: str = "harmfulness"  # or "refusal"
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract activations and labels from compressed data.

    Args:
        compressed_data: List of MinimalActs objects
        position: "inst" or "postinst"
        label_type: "harmfulness" (harmful/harmless) or "refusal" (refused/accepted)

    Returns:
        X: [n_samples, n_layers, d_model] activation array
        y: [n_samples] binary labels
    """
    n_samples = len(compressed_data)
    n_layers = len(compressed_data[0].acts_inst)
    d_model = compressed_data[0].acts_inst[0].shape[0]

    X = np.zeros((n_samples, n_layers, d_model), dtype=np.float32)
    y = np.zeros(n_samples, dtype=int)

    for i, example in enumerate(compressed_data):
        # Get activations for specified position
        acts = example.acts_inst if position == "inst" else example.acts_postinst

        # Stack all layers
        for layer_idx in range(n_layers):
            X[i, layer_idx, :] = acts[layer_idx]

        # Get label
        if label_type == "harmfulness":
            y[i] = 1 if example.label == "harmful" else 0
        else:  # refusal
            y[i] = 1 if example.refused else 0

    return X, y


def train_probe_single_layer(
    X_train: np.ndarray,  # [n_train, d_model]
    y_train: np.ndarray,  # [n_train]
    X_test: np.ndarray,   # [n_test, d_model]
    y_test: np.ndarray,   # [n_test]
    C: float = 1.0
) -> Dict:
    """Train a logistic regression probe on a single layer.

    Returns dict with:
        - direction: The probe weight vector
        - train_acc, test_acc: Accuracies
        - auc: Test set AUC
    """
    # Train probe
    probe = LogisticRegression(
        C=C,
        max_iter=1000,
        solver='liblinear',
        random_state=42
    )
    probe.fit(X_train, y_train)

    # Evaluate
    train_pred = probe.predict(X_train)
    test_pred = probe.predict(X_test)
    test_proba = probe.predict_proba(X_test)[:, 1]

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    auc = roc_auc_score(y_test, test_proba)

    # Extract direction (coefficients)
    direction = probe.coef_[0]  # [d_model]

    return {
        'direction': direction,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'auc': auc,
        'probe': probe
    }


def train_layerwise_probes(
    compressed_data: List,
    position: str = "inst",
    concept: str = "harmfulness",
    test_size: float = 0.3,
    C: float = 1.0
) -> List[ProbeResult]:
    """Train probes for all layers.

    Args:
        compressed_data: List of MinimalActs objects
        position: "inst" or "postinst"
        concept: "harmfulness" or "refusal"
        test_size: Train/test split ratio
        C: Regularization parameter

    Returns:
        List of ProbeResult objects, one per layer
    """
    print(f"\n{'='*80}")
    print(f"Training {concept} probes at position '{position}'")
    print(f"{'='*80}")

    # Extract activations
    X, y = extract_activations_by_label(compressed_data, position, concept)
    n_samples, n_layers, d_model = X.shape

    print(f"Data shape: {X.shape}")
    print(f"Label distribution: {np.bincount(y)}")

    # Split data
    indices = np.arange(n_samples)
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    results = []

    # Train probe for each layer
    for layer in tqdm(range(n_layers), desc="Training probes"):
        X_layer_train = X[train_idx, layer, :]
        X_layer_test = X[test_idx, layer, :]
        y_train = y[train_idx]
        y_test = y[test_idx]

        # Train
        probe_result = train_probe_single_layer(
            X_layer_train, y_train,
            X_layer_test, y_test,
            C=C
        )

        # Store results
        results.append(ProbeResult(
            layer=layer,
            position=position,
            concept=concept,
            accuracy=probe_result['test_acc'],
            auc=probe_result['auc'],
            direction=probe_result['direction'],
            train_acc=probe_result['train_acc'],
            test_acc=probe_result['test_acc'],
            n_samples=n_samples
        ))

    # Print summary
    best_layer = max(results, key=lambda r: r.test_acc)
    print(f"\nBest layer: {best_layer.layer} (acc={best_layer.test_acc:.3f}, auc={best_layer.auc:.3f})")

    return results


# ============================================================================
# Clustering and Separation Metrics
# ============================================================================

def compute_silhouette_scores(
    compressed_data: List,
    position: str = "inst",
    label_type: str = "harmfulness"
) -> np.ndarray:
    """Compute silhouette scores for each layer.

    Higher scores indicate better separation between harmful/harmless clusters.

    Returns:
        scores: [n_layers] array of silhouette scores
    """
    X, y = extract_activations_by_label(compressed_data, position, label_type)
    n_layers = X.shape[1]

    scores = np.zeros(n_layers)

    for layer in range(n_layers):
        X_layer = X[:, layer, :]

        # Skip if only one class
        if len(np.unique(y)) < 2:
            scores[layer] = 0.0
            continue

        try:
            score = silhouette_score(X_layer, y, metric='cosine')
            scores[layer] = score
        except:
            scores[layer] = 0.0

    return scores


def compute_direction_similarity_matrix(directions: List[np.ndarray]) -> np.ndarray:
    """Compute cosine similarity between direction vectors across layers.

    Args:
        directions: List of [d_model] direction vectors

    Returns:
        similarity_matrix: [n_layers, n_layers] cosine similarity matrix
    """
    n_layers = len(directions)
    similarity = np.zeros((n_layers, n_layers))

    for i in range(n_layers):
        for j in range(n_layers):
            # Cosine similarity
            cos_sim = np.dot(directions[i], directions[j]) / (
                np.linalg.norm(directions[i]) * np.linalg.norm(directions[j])
            )
            similarity[i, j] = cos_sim

    return similarity


# ============================================================================
# Comprehensive Layer-wise Analysis
# ============================================================================

def analyze_layerwise_dynamics(
    compressed_data: List,
    position: str = "inst",
    test_size: float = 0.3,
    C: float = 1.0
) -> LayerWiseMetrics:
    """Run complete layer-wise analysis for both harmfulness and refusal.

    This is the main function for Priority #1: Layer-wise dynamics.

    Returns:
        LayerWiseMetrics object with all computed metrics
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE LAYER-WISE ANALYSIS")
    print("="*80)

    # Train probes for harmfulness
    harm_probes_inst = train_layerwise_probes(
        compressed_data, position, "harmfulness", test_size, C
    )

    # Train probes for refusal
    ref_probes_inst = train_layerwise_probes(
        compressed_data, position, "refusal", test_size, C
    )

    # Extract metrics
    n_layers = len(harm_probes_inst)
    layers = list(range(n_layers))

    harm_acc = np.array([p.accuracy for p in harm_probes_inst])
    ref_acc = np.array([p.accuracy for p in ref_probes_inst])
    harm_auc = np.array([p.auc for p in harm_probes_inst])
    ref_auc = np.array([p.auc for p in ref_probes_inst])

    harm_directions = [p.direction for p in harm_probes_inst]
    ref_directions = [p.direction for p in ref_probes_inst]

    # Compute silhouette scores
    print("\nComputing cluster separation metrics...")
    harm_silhouette = compute_silhouette_scores(compressed_data, position, "harmfulness")
    ref_silhouette = compute_silhouette_scores(compressed_data, position, "refusal")

    # Compute direction similarity matrices
    print("Computing direction similarity across layers...")
    harm_similarity = compute_direction_similarity_matrix(harm_directions)
    ref_similarity = compute_direction_similarity_matrix(ref_directions)

    metrics = LayerWiseMetrics(
        layers=layers,
        harmfulness_acc=harm_acc,
        refusal_acc=ref_acc,
        harmfulness_auc=harm_auc,
        refusal_auc=ref_auc,
        harmfulness_silhouette=harm_silhouette,
        refusal_silhouette=ref_silhouette,
        harmfulness_directions=harm_directions,
        refusal_directions=ref_directions,
        harmfulness_direction_similarity=harm_similarity,
        refusal_direction_similarity=ref_similarity
    )

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    best_harm_layer = np.argmax(harm_acc)
    best_ref_layer = np.argmax(ref_acc)

    print(f"\nBest harmfulness layer: {best_harm_layer}")
    print(f"  Accuracy: {harm_acc[best_harm_layer]:.3f}")
    print(f"  AUC: {harm_auc[best_harm_layer]:.3f}")
    print(f"  Silhouette: {harm_silhouette[best_harm_layer]:.3f}")

    print(f"\nBest refusal layer: {best_ref_layer}")
    print(f"  Accuracy: {ref_acc[best_ref_layer]:.3f}")
    print(f"  AUC: {ref_auc[best_ref_layer]:.3f}")
    print(f"  Silhouette: {ref_silhouette[best_ref_layer]:.3f}")

    return metrics


# ============================================================================
# Category-Specific Analysis
# ============================================================================

# AdvBench harm categories (manually curated based on content)
HARM_CATEGORIES = {
    'violence': [
        'kill', 'murder', 'assault', 'attack', 'weapon', 'bomb', 'explosive',
        'shoot', 'stab', 'hurt', 'harm', 'torture', 'beat'
    ],
    'illegal': [
        'steal', 'hack', 'fraud', 'scam', 'counterfeit', 'forge', 'illegal',
        'smuggle', 'traffic', 'launder', 'bribe', 'extort'
    ],
    'hate_speech': [
        'racist', 'sexist', 'homophobic', 'slur', 'discriminate', 'hate',
        'offensive', 'derogatory', 'insult'
    ],
    'privacy': [
        'doxx', 'stalk', 'spy', 'surveillance', 'private', 'personal information',
        'identity theft', 'track'
    ],
    'misinformation': [
        'fake news', 'conspiracy', 'hoax', 'misleading', 'false', 'propaganda',
        'disinformation'
    ],
    'dangerous': [
        'poison', 'drug', 'suicide', 'self-harm', 'overdose', 'unsafe'
    ]
}


def categorize_harmful_example(text: str) -> str:
    """Categorize a harmful instruction based on keywords.

    Returns the category name or 'other' if no clear match.
    """
    text_lower = text.lower()

    category_scores = {}
    for category, keywords in HARM_CATEGORIES.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        category_scores[category] = score

    # Return category with highest score, or 'other'
    if max(category_scores.values()) > 0:
        return max(category_scores.items(), key=lambda x: x[1])[0]
    else:
        return 'other'


def split_by_category(compressed_data: List) -> Dict[str, List]:
    """Split harmful examples into categories.

    Returns dict mapping category name to list of MinimalActs objects.
    """
    categories = {cat: [] for cat in HARM_CATEGORIES.keys()}
    categories['other'] = []
    categories['harmless'] = []

    for example in compressed_data:
        if example.label == "harmless":
            categories['harmless'].append(example)
        else:
            # Find category
            category = categorize_harmful_example(
                # We need the original text - this is a limitation
                # For now, use the example_id or add metadata
                example.example_id  # Placeholder
            )
            categories[category].append(example)

    return categories


def analyze_category_specificity(
    compressed_data: List,
    layer: int,
    position: str = "inst",
    test_size: float = 0.3
) -> CategoryAnalysis:
    """Analyze category-specific harmfulness representations.

    This implements Priority #3: Category-specific analysis.

    For each category:
    1. Train a probe on that category vs harmless
    2. Test on other categories (cross-category generalization)
    3. Compute similarity between category directions

    Args:
        compressed_data: Full dataset
        layer: Which layer to analyze
        position: "inst" or "postinst"
        test_size: Train/test split ratio

    Returns:
        CategoryAnalysis object
    """
    print(f"\n{'='*80}")
    print(f"CATEGORY-SPECIFIC ANALYSIS (Layer {layer})")
    print(f"{'='*80}")

    # First, we need to add category labels to our data
    # This requires access to original text - let's create a simpler version
    # that works with what we have

    # For demonstration, we'll create synthetic categories based on
    # clustering of the harmful examples

    # Extract harmful examples only
    harmful_data = [ex for ex in compressed_data if ex.label == "harmful"]
    harmless_data = [ex for ex in compressed_data if ex.label == "harmless"]

    if len(harmful_data) < 50:
        print("⚠ Not enough harmful examples for category analysis")
        return None

    # Extract activations for harmful examples
    X_harmful, _ = extract_activations_by_label(harmful_data, position, "harmfulness")
    X_harmful_layer = X_harmful[:, layer, :]  # [n_harmful, d_model]

    # Use K-means to discover categories
    from sklearn.cluster import KMeans
    n_categories = min(4, len(harmful_data) // 20)  # At least 20 examples per category

    if n_categories < 2:
        print("⚠ Not enough data for clustering")
        return None

    print(f"\nClustering harmful examples into {n_categories} categories...")
    kmeans = KMeans(n_clusters=n_categories, random_state=42, n_init=10)
    category_labels = kmeans.fit_predict(X_harmful_layer)

    # Train probe for each category
    category_names = [f"category_{i}" for i in range(n_categories)]
    category_directions = {}
    generalization_matrix = np.zeros((n_categories, n_categories))

    for i, cat_name in enumerate(category_names):
        print(f"\nTraining probe for {cat_name}...")

        # Get examples for this category
        cat_indices = np.where(category_labels == i)[0]
        cat_data = [harmful_data[idx] for idx in cat_indices]

        # Combine with harmless for binary classification
        combined_data = cat_data + harmless_data

        # Extract activations
        X_cat, y_cat = extract_activations_by_label(combined_data, position, "harmfulness")
        X_cat_layer = X_cat[:, layer, :]

        # Train probe
        train_idx, test_idx = train_test_split(
            np.arange(len(combined_data)),
            test_size=test_size,
            random_state=42,
            stratify=y_cat
        )

        probe_result = train_probe_single_layer(
            X_cat_layer[train_idx], y_cat[train_idx],
            X_cat_layer[test_idx], y_cat[test_idx]
        )

        category_directions[cat_name] = probe_result['direction']

        print(f"  {cat_name}: {len(cat_data)} examples, acc={probe_result['test_acc']:.3f}")

        # Test on other categories (cross-generalization)
        for j in range(n_categories):
            other_indices = np.where(category_labels == j)[0]
            other_data = [harmful_data[idx] for idx in other_indices]
            other_combined = other_data + harmless_data

            X_other, y_other = extract_activations_by_label(other_combined, position, "harmfulness")
            X_other_layer = X_other[:, layer, :]

            # Test with category i's probe
            pred = probe_result['probe'].predict(X_other_layer)
            acc = accuracy_score(y_other, pred)
            generalization_matrix[i, j] = acc

    # Compute similarity matrix
    n_cat = len(category_names)
    similarity_matrix = np.zeros((n_cat, n_cat))

    for i in range(n_cat):
        for j in range(n_cat):
            dir_i = category_directions[category_names[i]]
            dir_j = category_directions[category_names[j]]
            similarity_matrix[i, j] = np.dot(dir_i, dir_j) / (
                np.linalg.norm(dir_i) * np.linalg.norm(dir_j)
            )

    analysis = CategoryAnalysis(
        categories=category_names,
        category_directions=category_directions,
        similarity_matrix=similarity_matrix,
        generalization_matrix=generalization_matrix,
        layer_specificity=None  # Would need to compute across all layers
    )

    print("\n" + "="*80)
    print("CATEGORY SIMILARITY MATRIX")
    print("="*80)
    print(pd.DataFrame(
        similarity_matrix,
        index=category_names,
        columns=category_names
    ).round(3))

    print("\n" + "="*80)
    print("CROSS-CATEGORY GENERALIZATION")
    print("="*80)
    print("Rows = trained on, Cols = tested on")
    print(pd.DataFrame(
        generalization_matrix,
        index=category_names,
        columns=category_names
    ).round(3))

    return analysis


# ============================================================================
# Compositional Harmfulness
# ============================================================================

@dataclass
class CompositionalExample:
    """Example with compositional harm signal."""
    base_instruction: str
    context_frame: str
    full_instruction: str
    expected_harm_level: str  # "high", "medium", "low", "none"


def create_compositional_examples(harmful_instructions: List[str]) -> List[CompositionalExample]:
    """Create compositional harmfulness examples.

    This implements Priority #2: Compositional harmfulness.

    Takes base harmful instructions and wraps them in different contexts
    to see how the model's harmfulness representation changes.
    """
    contexts = {
        'fiction': ("Write a fictional story about ", "medium"),
        'educational': ("Explain why someone might ", "low"),
        'research': (" for research purposes", "medium"),
        'prevention': ("How to prevent ", "none"),
        'raw': ("", "high"),  # No context
        'roleplay': ("In a hypothetical scenario, ", "medium"),
    }

    examples = []

    for base in harmful_instructions[:10]:  # Limit for testing
        for frame_name, (frame_template, harm_level) in contexts.items():
            if frame_name == 'prevention':
                full = frame_template + base
            elif frame_name == 'raw':
                full = base
            else:
                full = frame_template + base

            examples.append(CompositionalExample(
                base_instruction=base,
                context_frame=frame_name,
                full_instruction=full,
                expected_harm_level=harm_level
            ))

    return examples


# ============================================================================
# Visualization
# ============================================================================

def plot_layerwise_metrics(metrics: LayerWiseMetrics, save_path: Optional[str] = None):
    """Create comprehensive visualization of layer-wise dynamics.

    Generates a multi-panel figure showing:
    1. Probe accuracy by layer
    2. AUC scores by layer
    3. Silhouette scores by layer
    4. Direction similarity heatmaps
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    layers = metrics.layers

    # Panel 1: Probe Accuracy
    ax = axes[0, 0]
    ax.plot(layers, metrics.harmfulness_acc, 'o-', label='Harmfulness', linewidth=2)
    ax.plot(layers, metrics.refusal_acc, 's-', label='Refusal', linewidth=2)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Probe Accuracy')
    ax.set_title('Probe Performance by Layer')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: AUC Scores
    ax = axes[0, 1]
    ax.plot(layers, metrics.harmfulness_auc, 'o-', label='Harmfulness', linewidth=2)
    ax.plot(layers, metrics.refusal_auc, 's-', label='Refusal', linewidth=2)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax.set_xlabel('Layer')
    ax.set_ylabel('AUC')
    ax.set_title('AUC by Layer')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Silhouette Scores
    ax = axes[0, 2]
    ax.plot(layers, metrics.harmfulness_silhouette, 'o-', label='Harmfulness', linewidth=2)
    ax.plot(layers, metrics.refusal_silhouette, 's-', label='Refusal', linewidth=2)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Cluster Separation by Layer')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 4: Harmfulness Direction Similarity
    ax = axes[1, 0]
    im = ax.imshow(metrics.harmfulness_direction_similarity, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Layer')
    ax.set_title('Harmfulness Direction Similarity')
    plt.colorbar(im, ax=ax)

    # Panel 5: Refusal Direction Similarity
    ax = axes[1, 1]
    im = ax.imshow(metrics.refusal_direction_similarity, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Layer')
    ax.set_title('Refusal Direction Similarity')
    plt.colorbar(im, ax=ax)

    # Panel 6: Summary stats
    ax = axes[1, 2]
    ax.axis('off')

    # Find critical layers
    best_harm_layer = layers[np.argmax(metrics.harmfulness_acc)]
    best_ref_layer = layers[np.argmax(metrics.refusal_acc)]

    # Compute "crystallization point" - first layer with >70% accuracy
    harm_crystal = next((l for l in layers if metrics.harmfulness_acc[l] > 0.7), -1)
    ref_crystal = next((l for l in layers if metrics.refusal_acc[l] > 0.7), -1)

    summary_text = f"""
    CRITICAL LAYERS

    Best Harmfulness: Layer {best_harm_layer}
      Accuracy: {metrics.harmfulness_acc[best_harm_layer]:.3f}
      AUC: {metrics.harmfulness_auc[best_harm_layer]:.3f}
      Silhouette: {metrics.harmfulness_silhouette[best_harm_layer]:.3f}

    Best Refusal: Layer {best_ref_layer}
      Accuracy: {metrics.refusal_acc[best_ref_layer]:.3f}
      AUC: {metrics.refusal_auc[best_ref_layer]:.3f}
      Silhouette: {metrics.refusal_silhouette[best_ref_layer]:.3f}

    Crystallization Points:
      Harmfulness: Layer {harm_crystal if harm_crystal != -1 else 'N/A'}
      Refusal: Layer {ref_crystal if ref_crystal != -1 else 'N/A'}
    """

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved figure to {save_path}")

    plt.show()


def plot_category_analysis(analysis: CategoryAnalysis, save_path: Optional[str] = None):
    """Visualize category-specific analysis results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Similarity matrix
    ax = axes[0]
    im = ax.imshow(analysis.similarity_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(analysis.categories)))
    ax.set_yticks(range(len(analysis.categories)))
    ax.set_xticklabels(analysis.categories, rotation=45, ha='right')
    ax.set_yticklabels(analysis.categories)
    ax.set_title('Category Direction Similarity')
    plt.colorbar(im, ax=ax)

    # Add values
    for i in range(len(analysis.categories)):
        for j in range(len(analysis.categories)):
            text = ax.text(j, i, f'{analysis.similarity_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=9)

    # Panel 2: Generalization matrix
    ax = axes[1]
    im = ax.imshow(analysis.generalization_matrix, cmap='YlGn', vmin=0, vmax=1)
    ax.set_xticks(range(len(analysis.categories)))
    ax.set_yticks(range(len(analysis.categories)))
    ax.set_xticklabels(analysis.categories, rotation=45, ha='right')
    ax.set_yticklabels(analysis.categories)
    ax.set_title('Cross-Category Generalization\n(rows=trained, cols=tested)')
    plt.colorbar(im, ax=ax)

    # Add values
    for i in range(len(analysis.categories)):
        for j in range(len(analysis.categories)):
            text = ax.text(j, i, f'{analysis.generalization_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved figure to {save_path}")

    plt.show()


# ============================================================================
# Save/Load Analysis Results
# ============================================================================

def save_analysis(metrics: LayerWiseMetrics, filepath: str):
    """Save layer-wise analysis results."""
    with open(filepath, 'wb') as f:
        pickle.dump(metrics, f)
    print(f"✓ Saved analysis to {filepath}")


def load_analysis(filepath: str) -> LayerWiseMetrics:
    """Load layer-wise analysis results."""
    with open(filepath, 'rb') as f:
        metrics = pickle.load(f)
    print(f"✓ Loaded analysis from {filepath}")
    return metrics
