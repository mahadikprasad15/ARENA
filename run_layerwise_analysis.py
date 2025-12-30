#!/usr/bin/env python3
"""
End-to-end script for running layer-wise harmfulness/refusal analysis.

This script demonstrates a complete workflow from loading cached activations
to generating all analysis plots and results.

Usage:
    python run_layerwise_analysis.py --cache cache/llama3_1b_mixed_500.pkl

Requirements:
    - Cached activations file (generated from Harmfulness_and_Refusal_Probes.ipynb)
"""

import argparse
import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from layerwise_analysis import (
    analyze_layerwise_dynamics,
    analyze_category_specificity,
    plot_layerwise_metrics,
    plot_category_analysis,
    save_analysis,
)


def main():
    parser = argparse.ArgumentParser(description='Run layer-wise analysis')
    parser.add_argument(
        '--cache',
        type=str,
        default='cache/llama3_1b_mixed_500.pkl',
        help='Path to cached activations file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results_layerwise',
        help='Output directory for results'
    )
    parser.add_argument(
        '--position',
        type=str,
        default='inst',
        choices=['inst', 'postinst'],
        help='Which position to analyze'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.3,
        help='Train/test split ratio'
    )
    parser.add_argument(
        '--skip-category',
        action='store_true',
        help='Skip category analysis (faster)'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    (output_dir / 'figures').mkdir(exist_ok=True)

    print("="*80)
    print("LAYER-WISE HARMFULNESS AND REFUSAL ANALYSIS")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Cache file: {args.cache}")
    print(f"  Output directory: {output_dir}")
    print(f"  Position: {args.position}")
    print(f"  Test size: {args.test_size}")
    print(f"  Skip category analysis: {args.skip_category}")

    # Load cached activations
    print(f"\n{'='*80}")
    print("LOADING DATA")
    print("="*80)

    cache_path = Path(args.cache)
    if not cache_path.exists():
        print(f"❌ Error: Cache file not found: {cache_path}")
        print(f"\nPlease run Harmfulness_and_Refusal_Probes.ipynb first to generate cached activations.")
        return

    with open(cache_path, 'rb') as f:
        compressed_data = pickle.load(f)

    print(f"✓ Loaded {len(compressed_data)} examples")

    # Print data statistics
    n_harmful = sum(1 for ex in compressed_data if ex.label == "harmful")
    n_harmless = len(compressed_data) - n_harmful
    n_refused = sum(1 for ex in compressed_data if ex.refused)

    print(f"\nDataset composition:")
    print(f"  Harmful: {n_harmful} ({n_harmful/len(compressed_data)*100:.1f}%)")
    print(f"  Harmless: {n_harmless} ({n_harmless/len(compressed_data)*100:.1f}%)")
    print(f"  Refused: {n_refused} ({n_refused/len(compressed_data)*100:.1f}%)")

    # Run layer-wise analysis
    print(f"\n{'='*80}")
    print("PRIORITY #1: LAYER-WISE DYNAMICS")
    print("="*80)

    metrics = analyze_layerwise_dynamics(
        compressed_data,
        position=args.position,
        test_size=args.test_size,
        C=1.0
    )

    # Save metrics
    metrics_path = output_dir / f'layerwise_metrics_{args.position}.pkl'
    save_analysis(metrics, str(metrics_path))

    # Generate visualizations
    print(f"\nGenerating visualizations...")
    plot_layerwise_metrics(
        metrics,
        save_path=str(output_dir / 'figures' / f'layerwise_dynamics_{args.position}.png')
    )

    # Print summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print("="*80)

    best_harm_layer = np.argmax(metrics.harmfulness_acc)
    best_ref_layer = np.argmax(metrics.refusal_acc)

    print(f"\nBest Harmfulness Layer: {best_harm_layer}")
    print(f"  Accuracy: {metrics.harmfulness_acc[best_harm_layer]:.3f}")
    print(f"  AUC: {metrics.harmfulness_auc[best_harm_layer]:.3f}")
    print(f"  Silhouette: {metrics.harmfulness_silhouette[best_harm_layer]:.3f}")

    print(f"\nBest Refusal Layer: {best_ref_layer}")
    print(f"  Accuracy: {metrics.refusal_acc[best_ref_layer]:.3f}")
    print(f"  AUC: {metrics.refusal_auc[best_ref_layer]:.3f}")
    print(f"  Silhouette: {metrics.refusal_silhouette[best_ref_layer]:.3f}")

    # Crystallization points
    print(f"\nCrystallization Points (70% accuracy threshold):")
    harm_crystal = next((i for i, acc in enumerate(metrics.harmfulness_acc) if acc > 0.7), -1)
    ref_crystal = next((i for i, acc in enumerate(metrics.refusal_acc) if acc > 0.7), -1)

    if harm_crystal != -1:
        print(f"  Harmfulness: Layer {harm_crystal}")
    else:
        print(f"  Harmfulness: Not reached")

    if ref_crystal != -1:
        print(f"  Refusal: Layer {ref_crystal}")
    else:
        print(f"  Refusal: Not reached")

    # Analyze layer groups
    n_layers = len(metrics.layers)
    early = range(0, n_layers // 3)
    middle = range(n_layers // 3, 2 * n_layers // 3)
    late = range(2 * n_layers // 3, n_layers)

    print(f"\nLayer Group Analysis:")
    print(f"  Early layers (0-{n_layers//3-1}):")
    print(f"    Harmfulness mean acc: {np.mean([metrics.harmfulness_acc[i] for i in early]):.3f}")
    print(f"    Refusal mean acc: {np.mean([metrics.refusal_acc[i] for i in early]):.3f}")

    print(f"  Middle layers ({n_layers//3}-{2*n_layers//3-1}):")
    print(f"    Harmfulness mean acc: {np.mean([metrics.harmfulness_acc[i] for i in middle]):.3f}")
    print(f"    Refusal mean acc: {np.mean([metrics.refusal_acc[i] for i in middle]):.3f}")

    print(f"  Late layers ({2*n_layers//3}-{n_layers-1}):")
    print(f"    Harmfulness mean acc: {np.mean([metrics.harmfulness_acc[i] for i in late]):.3f}")
    print(f"    Refusal mean acc: {np.mean([metrics.refusal_acc[i] for i in late]):.3f}")

    # Category analysis
    if not args.skip_category:
        print(f"\n{'='*80}")
        print("PRIORITY #3: CATEGORY-SPECIFIC ANALYSIS")
        print("="*80)

        print(f"\nAnalyzing category specificity at layer {best_harm_layer}...")

        cat_analysis = analyze_category_specificity(
            compressed_data,
            layer=best_harm_layer,
            position=args.position,
            test_size=args.test_size
        )

        if cat_analysis is not None:
            # Save results
            cat_path = output_dir / 'category_analysis.pkl'
            with open(cat_path, 'wb') as f:
                pickle.dump(cat_analysis, f)
            print(f"✓ Saved category analysis to {cat_path}")

            # Visualize
            plot_category_analysis(
                cat_analysis,
                save_path=str(output_dir / 'figures' / 'category_analysis.png')
            )

            # Print interpretation
            print(f"\nCategory Analysis Interpretation:")
            avg_off_diag_sim = np.mean([
                cat_analysis.similarity_matrix[i, j]
                for i in range(len(cat_analysis.categories))
                for j in range(len(cat_analysis.categories))
                if i != j
            ])
            avg_off_diag_gen = np.mean([
                cat_analysis.generalization_matrix[i, j]
                for i in range(len(cat_analysis.categories))
                for j in range(len(cat_analysis.categories))
                if i != j
            ])

            print(f"  Average off-diagonal similarity: {avg_off_diag_sim:.3f}")
            print(f"  Average off-diagonal generalization: {avg_off_diag_gen:.3f}")

            if avg_off_diag_sim > 0.8 and avg_off_diag_gen > 0.7:
                print(f"\n  → UNIVERSAL representation (high similarity + high generalization)")
            elif avg_off_diag_sim < 0.5 and avg_off_diag_gen < 0.6:
                print(f"\n  → CATEGORY-SPECIFIC representations (low similarity + low generalization)")
            else:
                print(f"\n  → MIXED pattern (partial overlap between categories)")

    # Final summary
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Metrics: {metrics_path}")
    print(f"  - Figures: {output_dir / 'figures'}")

    if not args.skip_category and cat_analysis is not None:
        print(f"  - Category analysis: {output_dir / 'category_analysis.pkl'}")

    print(f"\nNext steps:")
    print(f"  1. Review the figures in {output_dir / 'figures'}")
    print(f"  2. Run notebook for interactive exploration")
    print(f"  3. Test compositional harmfulness (see notebook)")
    print(f"  4. Compare with t_postinst position (run with --position postinst)")


if __name__ == '__main__':
    main()
