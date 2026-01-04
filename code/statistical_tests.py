#!/usr/bin/env python3
"""
Statistical Significance Testing for FL Experiments
====================================================

Provides statistical tests for comparing federated learning methods,
including paired t-tests, effect size calculation, and confidence intervals.

Required for TMLR submission to demonstrate statistical rigor.

Author: Almas Ospanov
License: MIT
"""

import numpy as np
from scipy import stats
from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def paired_t_test(
    results_a: List[float],
    results_b: List[float],
    alpha: float = 0.05,
    method_a_name: str = "Method A",
    method_b_name: str = "Method B"
) -> Dict:
    """
    Perform paired t-test between two methods.

    Args:
        results_a: Results from method A (e.g., accuracies across seeds)
        results_b: Results from method B
        alpha: Significance level
        method_a_name: Name of method A
        method_b_name: Name of method B

    Returns:
        Dictionary with test results
    """
    if len(results_a) != len(results_b):
        raise ValueError("Results must have same length")

    if len(results_a) < 2:
        raise ValueError("Need at least 2 samples for t-test")

    # Paired t-test
    t_statistic, p_value = stats.ttest_rel(results_a, results_b)

    # Effect size (Cohen's d for paired samples)
    differences = np.array(results_a) - np.array(results_b)
    cohen_d = np.mean(differences) / np.std(differences, ddof=1)

    # Confidence interval for mean difference
    mean_diff = np.mean(differences)
    sem = stats.sem(differences)
    ci = stats.t.interval(1 - alpha, len(differences) - 1, loc=mean_diff, scale=sem)

    # Determine significance
    is_significant = p_value < alpha

    result = {
        'method_a': method_a_name,
        'method_b': method_b_name,
        'mean_a': float(np.mean(results_a)),
        'std_a': float(np.std(results_a, ddof=1)),
        'mean_b': float(np.mean(results_b)),
        'std_b': float(np.std(results_b, ddof=1)),
        'mean_difference': float(mean_diff),
        't_statistic': float(t_statistic),
        'p_value': float(p_value),
        'is_significant': bool(is_significant),
        'significance_level': alpha,
        'cohen_d': float(cohen_d),
        'effect_size_interpretation': interpret_cohen_d(cohen_d),
        'ci_lower': float(ci[0]),
        'ci_upper': float(ci[1]),
        'num_samples': len(results_a)
    }

    return result


def interpret_cohen_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05
) -> Tuple[List[float], List[bool]]:
    """
    Apply Bonferroni correction for multiple comparisons.

    Args:
        p_values: List of p-values
        alpha: Family-wise error rate

    Returns:
        (corrected_alpha, is_significant_list)
    """
    num_tests = len(p_values)
    corrected_alpha = alpha / num_tests

    is_significant = [p < corrected_alpha for p in p_values]

    return corrected_alpha, is_significant


def compute_confidence_interval(
    data: List[float],
    confidence: float = 0.95
) -> Tuple[float, Tuple[float, float]]:
    """
    Compute confidence interval for mean.

    Args:
        data: Sample data
        confidence: Confidence level (e.g., 0.95 for 95%)

    Returns:
        (mean, (lower_bound, upper_bound))
    """
    data_array = np.array(data)
    mean = np.mean(data_array)
    sem = stats.sem(data_array)
    ci = stats.t.interval(confidence, len(data_array) - 1, loc=mean, scale=sem)

    return float(mean), (float(ci[0]), float(ci[1]))


def wilcoxon_signed_rank_test(
    results_a: List[float],
    results_b: List[float],
    alpha: float = 0.05
) -> Dict:
    """
    Non-parametric alternative to paired t-test.

    Use when data may not be normally distributed.

    Args:
        results_a: Results from method A
        results_b: Results from method B
        alpha: Significance level

    Returns:
        Dictionary with test results
    """
    if len(results_a) != len(results_b):
        raise ValueError("Results must have same length")

    statistic, p_value = stats.wilcoxon(results_a, results_b)

    is_significant = p_value < alpha

    return {
        'test': 'Wilcoxon Signed-Rank',
        'statistic': float(statistic),
        'p_value': float(p_value),
        'is_significant': bool(is_significant),
        'significance_level': alpha,
        'median_a': float(np.median(results_a)),
        'median_b': float(np.median(results_b)),
        'median_difference': float(np.median(np.array(results_a) - np.array(results_b)))
    }


def mann_whitney_u_test(
    results_a: List[float],
    results_b: List[float],
    alpha: float = 0.05
) -> Dict:
    """
    Mann-Whitney U test for independent samples.

    Use when comparing two independent groups.

    Args:
        results_a: Results from method A
        results_b: Results from method B
        alpha: Significance level

    Returns:
        Dictionary with test results
    """
    statistic, p_value = stats.mannwhitneyu(results_a, results_b, alternative='two-sided')

    is_significant = p_value < alpha

    return {
        'test': 'Mann-Whitney U',
        'statistic': float(statistic),
        'p_value': float(p_value),
        'is_significant': bool(is_significant),
        'significance_level': alpha,
        'median_a': float(np.median(results_a)),
        'median_b': float(np.median(results_b))
    }


def compare_multiple_methods(
    results_dict: Dict[str, List[float]],
    baseline_method: str,
    alpha: float = 0.05,
    use_bonferroni: bool = True
) -> Dict:
    """
    Compare multiple methods against a baseline.

    Args:
        results_dict: Dictionary mapping method names to result lists
        baseline_method: Name of baseline method
        alpha: Significance level
        use_bonferroni: Whether to apply Bonferroni correction

    Returns:
        Dictionary with all pairwise comparison results
    """
    if baseline_method not in results_dict:
        raise ValueError(f"Baseline method '{baseline_method}' not in results")

    baseline_results = results_dict[baseline_method]
    comparisons = []

    # Perform pairwise tests
    for method_name, method_results in results_dict.items():
        if method_name == baseline_method:
            continue

        test_result = paired_t_test(
            method_results,
            baseline_results,
            alpha=alpha,
            method_a_name=method_name,
            method_b_name=baseline_method
        )
        comparisons.append(test_result)

    # Apply Bonferroni correction if requested
    if use_bonferroni and len(comparisons) > 0:
        p_values = [c['p_value'] for c in comparisons]
        corrected_alpha, is_sig_corrected = bonferroni_correction(p_values, alpha)

        for i, comp in enumerate(comparisons):
            comp['bonferroni_corrected_alpha'] = corrected_alpha
            comp['is_significant_bonferroni'] = is_sig_corrected[i]

    # Summary statistics
    summary = {
        'baseline_method': baseline_method,
        'num_comparisons': len(comparisons),
        'alpha': alpha,
        'bonferroni_correction_applied': use_bonferroni,
        'comparisons': comparisons
    }

    return summary


def generate_latex_table(comparison_results: Dict, output_file: Optional[str] = None) -> str:
    """
    Generate LaTeX table from comparison results.

    Args:
        comparison_results: Output from compare_multiple_methods
        output_file: Optional file to save LaTeX code

    Returns:
        LaTeX table string
    """
    baseline = comparison_results['baseline_method']
    comparisons = comparison_results['comparisons']

    # Start table
    latex = "\\begin{table}[t]\n"
    latex += "\\centering\n"
    latex += "\\caption{Statistical Comparison Against " + baseline.replace("_", "\\_") + "}\n"
    latex += "\\label{tab:statistical_comparison}\n"
    latex += "\\begin{tabular}{lccccl}\n"
    latex += "\\toprule\n"
    latex += "Method & Mean $\\pm$ Std & $\\Delta$ & $p$-value & Cohen's $d$ & Significant \\\\\n"
    latex += "\\midrule\n"

    # Baseline row
    latex += f"{baseline.replace('_', '\\_')} (baseline) & "
    if comparisons:
        baseline_mean = comparisons[0]['mean_b']
        baseline_std = comparisons[0]['std_b']
        latex += f"${baseline_mean:.2f} \\pm {baseline_std:.2f}$ & -- & -- & -- & -- \\\\\n"

    # Comparison rows
    for comp in comparisons:
        method = comp['method_a'].replace("_", "\\_")
        mean_a = comp['mean_a']
        std_a = comp['std_a']
        mean_diff = comp['mean_difference']
        p_val = comp['p_value']
        cohen_d = comp['cohen_d']

        # Determine significance marker
        use_bonferroni = 'bonferroni_corrected_alpha' in comp
        if use_bonferroni:
            sig_marker = "$^{***}$" if comp['is_significant_bonferroni'] else ""
        else:
            sig_marker = "$^{*}$" if comp['is_significant'] else ""

        latex += f"{method} & "
        latex += f"${mean_a:.2f} \\pm {std_a:.2f}$ & "
        latex += f"${mean_diff:+.2f}$ & "

        # Format p-value
        if p_val < 0.001:
            latex += "$<0.001$ & "
        else:
            latex += f"${p_val:.3f}$ & "

        latex += f"${cohen_d:.2f}$ & "
        latex += f"{comp['effect_size_interpretation'].capitalize()} {sig_marker} \\\\\n"

    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"

    # Add footnote
    if any('bonferroni_corrected_alpha' in c for c in comparisons):
        corrected_alpha = comparisons[0]['bonferroni_corrected_alpha']
        latex += f"\\\\[0.5em]\n"
        latex += f"\\small $^{{***}}$ Significant after Bonferroni correction ($\\alpha = {corrected_alpha:.4f}$).\n"
    else:
        latex += f"\\\\[0.5em]\n"
        latex += f"\\small $^{{*}}$ Significant at $\\alpha = {comparison_results['alpha']}$.\n"

    latex += "\\end{table}\n"

    if output_file:
        with open(output_file, 'w') as f:
            f.write(latex)
        logger.info(f"LaTeX table saved to {output_file}")

    return latex


def analyze_experimental_results(
    results_dir: str,
    baseline_method: str = 'fedavg',
    metric: str = 'final_accuracy',
    output_dir: Optional[str] = None
) -> Dict:
    """
    Analyze experimental results from JSON files.

    Args:
        results_dir: Directory containing result JSON files
        baseline_method: Name of baseline method
        metric: Metric to compare (e.g., 'final_accuracy')
        output_dir: Directory to save analysis outputs

    Returns:
        Dictionary with analysis results
    """
    results_dir = Path(results_dir)

    # Collect results
    results_dict = {}

    for json_file in results_dir.glob("*.json"):
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Extract results for each method
        if isinstance(data, dict):
            for method_name, method_data in data.items():
                if isinstance(method_data, dict) and metric in method_data:
                    if method_name not in results_dict:
                        results_dict[method_name] = []
                    results_dict[method_name].append(method_data[metric])

    if not results_dict:
        raise ValueError(f"No results found in {results_dir}")

    logger.info(f"Found results for {len(results_dict)} methods")
    for method, results in results_dict.items():
        logger.info(f"  {method}: {len(results)} runs")

    # Perform statistical comparison
    comparison_results = compare_multiple_methods(
        results_dict,
        baseline_method=baseline_method,
        alpha=0.05,
        use_bonferroni=True
    )

    # Generate LaTeX table
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        latex_file = output_dir / "statistical_comparison.tex"
        generate_latex_table(comparison_results, str(latex_file))

        # Save JSON results
        json_file = output_dir / "statistical_analysis.json"
        with open(json_file, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        logger.info(f"Analysis results saved to {json_file}")

    return comparison_results


def print_comparison_summary(comparison_results: Dict):
    """Print human-readable summary of comparison results."""
    print("\n" + "="*70)
    print("STATISTICAL COMPARISON SUMMARY")
    print("="*70)

    baseline = comparison_results['baseline_method']
    print(f"Baseline: {baseline}")
    print(f"Number of comparisons: {comparison_results['num_comparisons']}")
    print(f"Significance level: α = {comparison_results['alpha']}")
    print(f"Bonferroni correction: {'Yes' if comparison_results['bonferroni_correction_applied'] else 'No'}")

    if comparison_results['bonferroni_correction_applied']:
        corrected_alpha = comparison_results['comparisons'][0].get('bonferroni_corrected_alpha')
        print(f"Corrected significance level: α = {corrected_alpha:.4f}")

    print("\n" + "-"*70)
    print(f"{'Method':<20} {'Mean±Std':<15} {'Δ':<10} {'p-value':<10} {'Cohen's d':<12} {'Sig?'}")
    print("-"*70)

    # Baseline
    if comparison_results['comparisons']:
        baseline_mean = comparison_results['comparisons'][0]['mean_b']
        baseline_std = comparison_results['comparisons'][0]['std_b']
        print(f"{baseline:<20} {baseline_mean:.2f}±{baseline_std:.2f}")
        print("-"*70)

    # Comparisons
    for comp in comparison_results['comparisons']:
        method = comp['method_a']
        mean_a = comp['mean_a']
        std_a = comp['std_a']
        mean_diff = comp['mean_difference']
        p_val = comp['p_value']
        cohen_d = comp['cohen_d']

        use_bonferroni = 'is_significant_bonferroni' in comp
        is_sig = comp['is_significant_bonferroni'] if use_bonferroni else comp['is_significant']
        sig_marker = "***" if is_sig else ""

        p_val_str = f"<0.001" if p_val < 0.001 else f"{p_val:.3f}"

        print(f"{method:<20} {mean_a:.2f}±{std_a:.2f:<6} {mean_diff:+.2f}     "
              f"{p_val_str:<10} {cohen_d:.2f} ({comp['effect_size_interpretation']:<8})  {sig_marker}")

    print("="*70 + "\n")


# Example usage
if __name__ == "__main__":
    # Simulated experiment results (replace with actual data)
    np.random.seed(42)

    results = {
        'fedavg': [88.1, 88.4, 87.9, 88.2, 88.5],
        'krum': [85.2, 85.8, 85.1, 85.5, 85.3],
        'bulyan': [86.5, 86.8, 86.2, 86.9, 86.4],
        'dsain': [90.1, 90.5, 89.8, 90.3, 90.2]
    }

    # Compare all methods against FedAvg
    comparison = compare_multiple_methods(
        results,
        baseline_method='fedavg',
        alpha=0.05,
        use_bonferroni=True
    )

    # Print summary
    print_comparison_summary(comparison)

    # Generate LaTeX table
    latex_table = generate_latex_table(comparison)
    print("\nLaTeX Table:")
    print(latex_table)
