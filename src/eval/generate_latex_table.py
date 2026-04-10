import json
import argparse
import os

def generate_latex_metrics(metrics):
    baseline = metrics['baseline']
    ablated = metrics['ablated']
    
    latex = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{lcc}\n\\hline\n"
    latex += "Metric & Baseline & Ablated \\\\\n\\hline\n"
    latex += f"Perplexity $\\downarrow$ & {baseline['perplexity']:.2f} & {ablated['perplexity']:.2f} \\\\\n"
    latex += f"Avg Distinct-3 $\\uparrow$ & {baseline['avg_distinct_3']:.4f} & {ablated['avg_distinct_3']:.4f} \\\\\n"
    latex += f"Degeneration Rate $\\downarrow$ & {baseline['degeneration_rate']:.4f} & {ablated['degeneration_rate']:.4f} \\\\\n"
    latex += "\\hline\n\\end{tabular}\n\\caption{General Text Quality Metrics}\n\\end{table}\n"
    return latex

def generate_latex_judge(metrics):
    if 'llm_judge' not in metrics['baseline']:
        return "LLM Judge data not found."
        
    baseline = metrics['baseline']['llm_judge']
    ablated = metrics['ablated']['llm_judge']
    
    classes = [c for c in baseline.keys() if c not in ['Error', 'Error_Format']]
    
    latex = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{lcc}\n\\hline\n"
    latex += "Classification & Baseline Count & Ablated Count \\\\\n\\hline\n"
    for c in classes:
        latex += f"{c} & {baseline[c]} & {ablated[c]} \\\\\n"
    latex += "\\hline\n\\end{tabular}\n\\caption{LLM Judge Classifications}\n\\end{table}\n"
    return latex

def generate_latex_shifts(metrics):
    shifts = metrics['summary']['logprob_shifts']
    
    latex = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{lc}\n\\hline\n"
    latex += "Category & Logprob Shift \\\\\n\\hline\n"
    for cat, val in shifts.items():
        latex += f"{cat.replace('_', ' ').title()} & {val:.4f} \\\\\n"
    latex += "\\hline\n\\end{tabular}\n\\caption{Logprob Shifts by Category (Ablated - Baseline)}\n\\end{table}\n"
    return latex

def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX Table from Results")
    parser.add_argument("--input", type=str, default="results/unified_eval_results_judged.json", help="Path to JSON results")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found.")
        return
        
    with open(args.input, "r") as f:
        data = json.load(f)
        
    metrics = data.get('metrics', {})
    if not metrics:
        print("No metrics found.")
        return
        
    print("\n% --- LaTeX TABLES ---\n")
    print(generate_latex_metrics(metrics))
    print("\n")
    print(generate_latex_judge(metrics))
    print("\n")
    print(generate_latex_shifts(metrics))
    print("\n% --- END LaTeX TABLES ---\n")

if __name__ == "__main__":
    main()
