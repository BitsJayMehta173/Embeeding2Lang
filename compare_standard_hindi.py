"""
compare_standard_hindi.py

Compares our H4 CP-CLE model against the Facebook pretrained
FastText Hindi vectors (cc.hi.300) on the exact same eval_pairs.json.

Standard model: Facebook AI Research - cc.hi.300.vec
  Trained on: Hindi Wikipedia + CommonCrawl (~27M sentences)
  Dimensions: 300
  This is what is normally used as a "Hindi word embedding".

Our model (H4): 100-dim, CP-CLE aligned to Glove-100 English space.
"""

import os
import json
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import KeyedVectors
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

DATA_DIR    = "data"
MODELS_DIR  = "models"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

FACEBOOK_VEC_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.hi.300.vec.gz"
FACEBOOK_VEC_GZ  = os.path.join(MODELS_DIR, "cc.hi.300.vec.gz")
FACEBOOK_VEC     = os.path.join(MODELS_DIR, "cc.hi.300.vec")
FACEBOOK_KV      = os.path.join(MODELS_DIR, "cc_hi_300.kv")


# ── Download ──────────────────────────────────────────────────────────────────

def download_facebook_hindi():
    if os.path.exists(FACEBOOK_KV):
        print("Facebook Hindi vectors already cached as KeyedVectors.")
        return KeyedVectors.load(FACEBOOK_KV)

    if not os.path.exists(FACEBOOK_VEC_GZ):
        print(f"Downloading Facebook cc.hi.300.vec.gz (~1.3 GB)...")
        def progress(count, block, total):
            pct = count * block / total * 100
            print(f"\r  {min(pct, 100):.1f}%", end="", flush=True)
        urllib.request.urlretrieve(FACEBOOK_VEC_URL, FACEBOOK_VEC_GZ, reporthook=progress)
        print("\nDownload complete.")

    if not os.path.exists(FACEBOOK_VEC):
        print("Decompressing...")
        import gzip, shutil
        with gzip.open(FACEBOOK_VEC_GZ, 'rb') as f_in:
            with open(FACEBOOK_VEC, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print("Done.")

    print("Loading vectors into KeyedVectors (this takes ~2 min)...")
    kv = KeyedVectors.load_word2vec_format(FACEBOOK_VEC, binary=False)
    kv.save(FACEBOOK_KV)
    print(f"Saved to {FACEBOOK_KV} for fast reuse.")
    return kv


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(kv, eval_data, threshold=None):
    pos_sims = []
    neg_sims = []

    for item in eval_data:
        if item["type"] == "cross_lingual":
            continue
        w1, w2 = item["hi1"], item["hi2"]
        if w1 in kv and w2 in kv:
            sim = kv.similarity(w1, w2)
            if item["type"] == "positive":
                pos_sims.append(sim)
            elif item["type"] == "negative":
                neg_sims.append(sim)

    if not pos_sims or not neg_sims:
        return None

    pos_mean = np.mean(pos_sims)
    neg_mean = np.mean(neg_sims)
    gap      = pos_mean - neg_mean

    # Sweep thresholds if none given
    if threshold is None:
        best_f1, best_thresh = -1, 0.0
        for t in np.arange(0.1, 0.91, 0.01):
            y_true = [1]*len(pos_sims) + [0]*len(neg_sims)
            y_pred = [1 if s >= t else 0 for s in pos_sims + neg_sims]
            f = f1_score(y_true, y_pred, zero_division=0)
            if f > best_f1:
                best_f1    = f
                best_thresh = t
        threshold = best_thresh

    y_true = [1]*len(pos_sims) + [0]*len(neg_sims)
    y_pred = [1 if s >= threshold else 0 for s in pos_sims + neg_sims]

    return {
        "pos_mean":  round(pos_mean, 4),
        "neg_mean":  round(neg_mean, 4),
        "gap":       round(gap, 4),
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "threshold": round(threshold, 2),
        "pos_sims":  pos_sims,
        "neg_sims":  neg_sims,
        "coverage":  len(pos_sims) + len(neg_sims),
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_comparison(results, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Standard Hindi Embedding vs H4 CP-CLE\n(Same Evaluation Corpus)", 
                 fontsize=14, fontweight="bold")

    colors = {"Standard (Facebook cc.hi.300)": ("#e74c3c", "#3498db"),
              "Our H4 (CP-CLE Aligned)":       ("#27ae60", "#8e44ad")}

    for ax, (model_name, res) in zip(axes, results.items()):
        pos_c, neg_c = colors[model_name]
        sns.kdeplot(res["pos_sims"], ax=ax, label=f"Synonyms  (mean={res['pos_mean']:.3f})",
                    fill=True, color=pos_c, alpha=0.5)
        sns.kdeplot(res["neg_sims"], ax=ax, label=f"Random    (mean={res['neg_mean']:.3f})",
                    fill=True, color=neg_c, alpha=0.5)
        ax.axvline(res["threshold"], color="black", linestyle="--", linewidth=1.5,
                   label=f"Threshold = {res['threshold']}")
        ax.set_title(f"{model_name}\nGap={res['gap']:.3f}  F1={res['f1']:.3f}  Acc={res['accuracy']:.1%}")
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved -> {output_path}")


def plot_bar_comparison(results, output_path):
    metrics  = ["gap", "accuracy", "f1", "precision", "recall"]
    labels   = ["Sep. Gap", "Accuracy", "F1-Score", "Precision", "Recall"]
    models   = list(results.keys())
    x        = np.arange(len(labels))
    width    = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2,
                   [results[models[0]][m] for m in metrics], width,
                   label=models[0], color="#e74c3c", alpha=0.85)
    bars2 = ax.bar(x + width/2,
                   [results[models[1]][m] for m in metrics], width,
                   label=models[1], color="#27ae60", alpha=0.85)

    ax.set_title("Standard Hindi Embedding vs H4 CP-CLE — Metric Comparison\n(Same Evaluation Set, Same Word Pairs)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.bar_label(bars1, fmt="%.3f", padding=3, fontsize=9)
    ax.bar_label(bars2, fmt="%.3f", padding=3, fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Bar chart saved -> {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading evaluation data...")
    with open(os.path.join(DATA_DIR, "eval_pairs.json"), encoding="utf-8") as f:
        eval_data = json.load(f)

    # 1. Standard Facebook Hindi model
    fb_kv = download_facebook_hindi()

    print("\nEvaluating: Standard Facebook cc.hi.300 ...")
    fb_res = evaluate(fb_kv, eval_data)

    # 2. Our H4 model
    print("Evaluating: H4 CP-CLE ...")
    h4_kv  = KeyedVectors.load(os.path.join(MODELS_DIR, "H4_cp_cle.kv"))
    h4_res = evaluate(h4_kv, eval_data)

    results = {
        "Standard (Facebook cc.hi.300)": fb_res,
        "Our H4 (CP-CLE Aligned)":       h4_res,
    }

    # ── Print ─────────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("COMPARISON: Standard Hindi Embedding vs H4 CP-CLE")
    print("(Evaluated on identical word pairs from eval_pairs.json)")
    print("="*65)

    header = f"{'Metric':<22} {'Standard (cc.hi.300)':>22} {'H4 CP-CLE':>14} {'Winner':>10}"
    print(header)
    print("-"*65)

    metric_labels = [
        ("pos_mean",  "Synonym Sim. Mean"),
        ("neg_mean",  "Random Sim. Mean"),
        ("gap",       "Separation Gap"),
        ("accuracy",  "Accuracy"),
        ("f1",        "F1-Score"),
        ("precision", "Precision"),
        ("recall",    "Recall"),
        ("threshold", "Optimal Threshold"),
        ("coverage",  "Pairs Evaluated"),
    ]

    for key, label in metric_labels:
        fb_val = fb_res[key]
        h4_val = h4_res[key]
        if key == "neg_mean":
            winner = "H4" if h4_val < fb_val else "Standard"
        elif key in ("threshold", "coverage"):
            winner = "-"
        else:
            winner = "H4" if h4_val > fb_val else "Standard"
        print(f"  {label:<20} {str(fb_val):>22} {str(h4_val):>14} {winner:>10}")

    print()
    print("Model Details:")
    print(f"  Standard (cc.hi.300):  300-dim, trained on Hindi Wikipedia + CommonCrawl")
    print(f"  H4 (CP-CLE):           100-dim, pseudo-context corpus + GloVe alignment")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_comparison(results, os.path.join(RESULTS_DIR, "Standard_vs_H4_distributions.png"))
    plot_bar_comparison(results, os.path.join(RESULTS_DIR, "Standard_vs_H4_metrics.png"))

    print("\nDone. Check results/ for plots.")


if __name__ == "__main__":
    main()
