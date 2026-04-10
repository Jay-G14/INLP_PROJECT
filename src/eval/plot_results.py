import json
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_general_metrics(metrics, output_dir):
    """Plot Perplexity, Distinct-3, and Degeneration Rate."""
    baseline = metrics['baseline']
    ablated = metrics['ablated']
    
    categories = ['Perplexity', 'Avg Distinct-3', 'Degeneration Rate']
    baseline_values = [baseline['perplexity'], baseline['avg_distinct_3'], baseline['degeneration_rate']]
    ablated_values = [ablated['perplexity'], ablated['avg_distinct_3'], ablated['degeneration_rate']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, baseline_values, width, label='Baseline', color='#4C72B0')
    rects2 = ax.bar(x + width/2, ablated_values, width, label='Ablated', color='#C44E52')
    
    ax.set_ylabel('Value')
    ax.set_title('General Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    
    # Scale Y axis for readability if needed, but Perplexity is usually much larger than the others
    # So let's create two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Perplexity
    ax1.bar(['Baseline', 'Ablated'], [baseline['perplexity'], ablated['perplexity']], color=['#4C72B0', '#C44E52'])
    ax1.set_title('Perplexity (Lower is better)')
    ax1.set_ylabel('Perplexity')
    
    # Distinct-3 and Degeneration Rate
    cat2 = ['Avg Distinct-3', 'Degeneration Rate']
    b2 = [baseline['avg_distinct_3'], baseline['degeneration_rate']]
    a2 = [ablated['avg_distinct_3'], ablated['degeneration_rate']]
    x2 = np.arange(len(cat2))
    
    ax2.bar(x2 - width/2, b2, width, label='Baseline', color='#4C72B0')
    ax2.bar(x2 + width/2, a2, width, label='Ablated', color='#C44E52')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(cat2)
    ax2.set_title('Text Quality Metrics')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'general_metrics.png'), dpi=300)
    plt.close()
    print(f"Saved general_metrics.png to {output_dir}")

def plot_logprob_shifts(metrics, output_dir):
    """Plot logprob shifts for different categories."""
    shifts = metrics['summary']['logprob_shifts']
    
    categories = list(shifts.keys())
    values = list(shifts.values())
    
    # Sort by shift value
    sorted_pairs = sorted(zip(categories, values), key=lambda x: x[1])
    categories, values = zip(*sorted_pairs)
    
    plt.figure(figsize=(10, 6))
    colors = ['#C44E52' if v < 0 else '#55A868' for v in values]
    plt.barh(categories, values, color=colors)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.xlabel('Logprob Shift (Ablated - Baseline)')
    plt.title('Logprob Shift by Category')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'logprob_shifts.png'), dpi=300)
    plt.close()
    print(f"Saved logprob_shifts.png to {output_dir}")

def plot_llm_judge_distribution(metrics, output_dir):
    """Plot distribution of LLM Judge classifications."""
    if 'llm_judge' not in metrics['baseline']:
        print("LLM Judge data not found in metrics. Skipping.")
        return
        
    baseline_judge = metrics['baseline']['llm_judge']
    ablated_judge = metrics['ablated']['llm_judge']
    
    # Filter out Error and Error_Format if they are 0
    classes = [c for c in baseline_judge.keys() if c not in ['Error', 'Error_Format']]
    baseline_counts = [baseline_judge[c] for c in classes]
    ablated_counts = [ablated_judge[c] for c in classes]
    
    # Total counts for normalization (optional, but let's use raw counts)
    x = np.arange(len(classes))
    width = 0.35
    
    plt.figure(figsize=(12, 7))
    plt.bar(x - width/2, baseline_counts, width, label='Baseline', color='#4C72B0')
    plt.bar(x + width/2, ablated_counts, width, label='Ablated', color='#C44E52')
    
    plt.ylabel('Number of Samples')
    plt.title('LLM Judge Classification Distribution')
    plt.xticks(x, classes)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'llm_judge_distribution.png'), dpi=300)
    plt.close()
    print(f"Saved llm_judge_distribution.png to {output_dir}")

def plot_logprob_comparison(metrics, output_dir):
    """Plot Log-Probability Comparison across Domains matching the user's requested style."""
    baseline_probs = metrics['baseline']['logprobs']
    ablated_probs = metrics['ablated']['logprobs']
    
    domains = ['HP', 'Magic', 'Fantasy', 'Real World']
    baseline_values = [baseline_probs.get('hp', 0), baseline_probs.get('magic', 0), baseline_probs.get('fantasy', 0), baseline_probs.get('real_world', 0)]
    ablated_values = [ablated_probs.get('hp', 0), ablated_probs.get('magic', 0), ablated_probs.get('fantasy', 0), ablated_probs.get('real_world', 0)]
    
    x = np.arange(len(domains))
    width = 0.4
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Custom colors matching the image
    baseline_color = '#8EBAD9'
    ablated_color = '#D99081'
    
    ax.bar(x - width/2, baseline_values, width, label='Baseline', color=baseline_color)
    ax.bar(x + width/2, ablated_values, width, label='Ablated', color=ablated_color)
    
    ax.set_ylabel('Average Log-Probability')
    ax.set_xlabel('Domain')
    ax.set_title('Log-Probability Comparison across Domains', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    
    ax.legend(title='Condition', loc='lower left')
    ax.grid(axis='y', linestyle='-', alpha=0.7)
    
    # Add footer text
    footer_text = "A lower (more negative) log-probability indicates the model is less confident.\nThe HP domain shows a significant drop, while others remain stable."
    plt.figtext(0.5, -0.05, footer_text, ha="center", fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'logprob_comparison_styled.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved logprob_comparison_styled.png to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Plot Evaluation Results")
    parser.add_argument("--input", type=str, default="results/unified_eval_results_judged.json", help="Path to JSON results")
    parser.add_argument("--output_dir", type=str, default="results/plots", help="Directory to save plots")
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    try:
        with open(args.input, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {args.input}: {e}")
        return
        
    metrics = data.get('metrics', {})
    if not metrics:
        print("No metrics found in result file.")
        return
        
    # Use seaborn style for beauty
    try:
        sns.set_theme(style="whitegrid")
    except:
        plt.style.use('ggplot')
        
    plot_general_metrics(metrics, args.output_dir)
    plot_logprob_shifts(metrics, args.output_dir)
    plot_llm_judge_distribution(metrics, args.output_dir)
    plot_logprob_comparison(metrics, args.output_dir)
    
    print(f"\nAll plots saved to {args.output_dir}")

if __name__ == "__main__":
    main()
