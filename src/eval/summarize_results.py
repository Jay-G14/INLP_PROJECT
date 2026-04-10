import json
import argparse
from collections import Counter


MODEL_DEFAULTS = {
    "llama": {
        "results_dir": "results/llama",
    },
    "gemma": {
        "results_dir": "results/gemma",
    },
    "mistral": {
        "results_dir": "results/mistral",
    },
    "gpt2": {
        "results_dir": "results",
    },
}


def main():
    parser = argparse.ArgumentParser(description="Summarize LLM Judge Results")
    parser.add_argument(
        "--model_family",
        type=str,
        choices=["llama", "gemma", "mistral", "gpt2"],
        default="gpt2",
        help="Model family to resolve default judged-results path for.",
    )
    parser.add_argument("--input", type=str, default=None, help="Path to JSON results")
    args = parser.parse_args()

    if args.input is None:
        results_dir = MODEL_DEFAULTS[args.model_family]["results_dir"]
        args.input = f"{results_dir}/unified_eval_results_judged.json"

    try:
        with open(args.input, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {args.input}: {e}")
        return

    side_by_side = data.get("side_by_side", [])
    total_prompts = len(side_by_side)
    
    if total_prompts == 0:
        print("No prompts found in the JSON file.")
        return

    baseline_counter = Counter()
    ablated_counter = Counter()

    for item in side_by_side:
        baseline_counter[item.get("baseline_llm_judge", "Unknown")] += 1
        ablated_counter[item.get("ablated_llm_judge", "Unknown")] += 1

    print("=" * 50)
    print("LLM JUDGE EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Total evaluated pairs: {total_prompts}\n")

    print("Baseline Classifications:")
    for key, count in baseline_counter.most_common():
        percentage = (count / total_prompts) * 100
        print(f"  {key}: {count} ({percentage:.1f}%)")

    print("\nAblated Classifications:")
    for key, count in ablated_counter.most_common():
        percentage = (count / total_prompts) * 100
        print(f"  {key}: {count} ({percentage:.1f}%)")
    print("=" * 50)

if __name__ == "__main__":
    main()
