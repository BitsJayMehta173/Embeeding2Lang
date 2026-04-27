import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import FastText, KeyedVectors
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DATA_DIR = "data"
MODELS_DIR = "models"
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_eval_data():
    with open(os.path.join(DATA_DIR, "eval_pairs.json"), 'r', encoding='utf-8') as f:
        return json.load(f)

def evaluate_model(model, eval_data, is_kv=False):
    pos_sims = []
    neg_sims = []
    cross_sims = []
    
    wv = model if is_kv else model.wv
    
    for item in eval_data:
        w1, w2 = item["hi1"], item["hi2"]
        if item["type"] == "cross_lingual":
            w1, w2 = item["en1"], item["hi1"] # cross lingual: English to Hindi
            
        if w1 in wv and w2 in wv:
            sim = wv.similarity(w1, w2)
            if item["type"] == "positive":
                pos_sims.append(sim)
            elif item["type"] == "negative":
                neg_sims.append(sim)
            elif item["type"] == "cross_lingual":
                cross_sims.append(sim)
                
    return pos_sims, neg_sims, cross_sims

def threshold_sweep(pos_sims, neg_sims):
    # True labels: 1 for positive pairs, 0 for negative pairs
    y_true = [1] * len(pos_sims) + [0] * len(neg_sims)
    all_sims = pos_sims + neg_sims
    
    thresholds = np.arange(0.3, 0.81, 0.01)
    best_f1 = -1
    best_thresh = 0
    best_metrics = {}
    f1_curve = []
    
    for thresh in thresholds:
        y_pred = [1 if s >= thresh else 0 for s in all_sims]
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        f1_curve.append(f1)
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            best_metrics = {
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1": f1,
                "Optimal_Threshold": thresh
            }
            
    return best_metrics, thresholds, f1_curve

def plot_distributions(results):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (model_name, (pos, neg, cross)) in enumerate(results.items()):
        ax = axes[i]
        sns.kdeplot(pos, ax=ax, label="Positive", fill=True, color='blue')
        sns.kdeplot(neg, ax=ax, label="Negative", fill=True, color='red')
        ax.set_title(f"{model_name} Distribution")
        ax.set_xlabel("Cosine Similarity")
        ax.legend()
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Similarity_Distributions_H4.png"))
    plt.close()

def plot_f1_curves(f1_data):
    plt.figure(figsize=(10, 6))
    for model_name, (thresholds, f1_curve) in f1_data.items():
        plt.plot(thresholds, f1_curve, label=model_name)
    plt.title("Threshold vs F1 Score")
    plt.xlabel("Similarity Threshold")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "F1_vs_Threshold.png"))
    plt.close()

def main():
    print("Loading evaluation data...")
    eval_data = load_eval_data()
    
    models_config = {
        "H1_Translated": (os.path.join(MODELS_DIR, "H1_translated.bin"), False),
        "H2_Native": (os.path.join(MODELS_DIR, "H2_native.bin"), False),
        "H3_Pseudo": (os.path.join(MODELS_DIR, "H3_pseudo.bin"), False),
        "H4_CP_CLE": (os.path.join(MODELS_DIR, "H4_cp_cle.kv"), True)
    }
    
    results = {}
    f1_data = {}
    summary_data = []
    
    for model_name, (path, is_kv) in models_config.items():
        print(f"Evaluating {model_name}...")
        try:
            if is_kv:
                model = KeyedVectors.load(path)
            else:
                model = FastText.load(path)
        except Exception as e:
            print(f"Could not load {model_name}: {e}")
            continue
            
        pos, neg, cross = evaluate_model(model, eval_data, is_kv)
        results[model_name] = (pos, neg, cross)
        
        pos_mean = np.mean(pos) if pos else 0
        neg_mean = np.mean(neg) if neg else 0
        gap = pos_mean - neg_mean
        
        metrics, thresholds, f1_curve = threshold_sweep(pos, neg)
        f1_data[model_name] = (thresholds, f1_curve)
        
        summary_data.append({
            "Model": model_name,
            "Pos Mean": round(pos_mean, 3),
            "Neg Mean": round(neg_mean, 3),
            "Gap": round(gap, 3),
            "Accuracy": round(metrics.get("Accuracy", 0), 3),
            "F1": round(metrics.get("F1", 0), 3),
            "Optimal Thresh": round(metrics.get("Optimal_Threshold", 0), 2)
        })
        
    plot_distributions(results)
    plot_f1_curves(f1_data)
    
    df = pd.DataFrame(summary_data)
    df.to_csv(os.path.join(OUTPUT_DIR, "augmented_evaluation_summary.csv"), index=False)
    
    print("\n--- Augmented Evaluation Summary ---")
    print(df.to_string(index=False))
    
    # Ablation printout
    print("\n--- Ablation Study: H3 vs H4 ---")
    h3_row = df[df["Model"] == "H3_Pseudo"].iloc[0]
    h4_row = df[df["Model"] == "H4_CP_CLE"].iloc[0]
    
    print(f"Gap Improvement: {h3_row['Gap']} -> {h4_row['Gap']} ({(h4_row['Gap'] - h3_row['Gap']):.3f})")
    print(f"F1 Improvement: {h3_row['F1']} -> {h4_row['F1']} ({(h4_row['F1'] - h3_row['F1']):.3f})")
    print(f"Accuracy Improvement: {h3_row['Accuracy']} -> {h4_row['Accuracy']} ({(h4_row['Accuracy'] - h3_row['Accuracy']):.3f})")

if __name__ == "__main__":
    main()
