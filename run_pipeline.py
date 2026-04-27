import subprocess
import sys

def run_script(script_name):
    print(f"\n{'='*50}")
    print(f"Running {script_name}...")
    print(f"{'='*50}\n")
    
    result = subprocess.run([sys.executable, script_name])
    
    if result.returncode != 0:
        print(f"\nError: {script_name} failed with return code {result.returncode}")
        sys.exit(result.returncode)

def generate_report():
    print("\nGenerating Augmented Markdown Report...")
    with open("results/augmented_evaluation_summary.csv", "r") as f:
        csv_content = f.read()
        
    report = f"""# Cross-Lingual Embedding Evaluation Report (CP-CLE Augmented)

## Objective
Extend the pseudo-context model (H3) with a Correlation-Preserving Cross-Lingual Embedding (CP-CLE) objective (H4) and evaluate its effectiveness as a semantic validator.

## Summary of Results
```csv
{csv_content}
```

## Observations
- **H1-H3**: Previous baseline models.
- **H4 (CP-CLE)**: Fine-tuned H3 embeddings constrained to match English pairwise cosine similarities using PyTorch.

Please refer to the `results/Similarity_Distributions_H4.png` and `results/F1_vs_Threshold.png` plots for detailed metrics.
"""
    with open("results/final_report_v2.md", "w") as f:
        f.write(report)
    print("Report saved to results/final_report_v2.md")

if __name__ == "__main__":
    scripts = [
        "data_prep.py",
        "build_datasets.py",
        "build_eval_set.py",
        "train_embeddings.py",
        "cp_cle_optimizer.py",
        "evaluate_cp_cle.py"
    ]
    
    for script in scripts:
        run_script(script)
        
    generate_report()
    print("\nPipeline execution completed successfully.")
