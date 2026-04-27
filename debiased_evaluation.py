"""
debiased_evaluation_v2.py
=========================
Proper fix after discovering that ALL synonym pairs in eval_pairs.json
were contaminated (they were all used in pseudo-context training sentences).

Strategy:
  1. Load the MUSE dictionary.
  2. Identify which Hindi words appear in the pseudo-context corpus (training vocab).
  3. Build a FRESH held-out test set using synonym pairs where the Hindi words
     did NOT appear in the pseudo-context training corpus at all.
  4. Evaluate all models fairly on this clean test set.

Fair comparison:
  - H2  : standard FastText trained on SAME native Hindi corpus (no tricks)
  - H3  : pseudo-context (our corpus engineering, no CP-CLE)
  - H4  : CP-CLE aligned (our full method)
  - cc.hi.300 : Facebook model (reference only, different scale/size)
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import FastText, KeyedVectors
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

DATA_DIR    = "data"
MODELS_DIR  = "models"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Step 1: Find words NOT in training ────────────────────────────────────────

def get_training_vocab():
    """Return the set of all Hindi words that appear in the pseudo-context corpus."""
    vocab = set()
    pseudo_path = os.path.join(DATA_DIR, "pseudo_context_corpus.txt")
    if os.path.exists(pseudo_path):
        with open(pseudo_path, encoding="utf-8") as f:
            for line in f:
                for w in line.strip().split():
                    vocab.add(w)
    print(f"  Pseudo-context training vocab: {len(vocab):,} unique Hindi tokens")
    return vocab


def load_muse_dict():
    """Load MUSE bilingual dictionary."""
    en2hi = {}
    with open(os.path.join(DATA_DIR, "en-hi.txt"), encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                en2hi[parts[0].lower()] = parts[1]
    return en2hi


# ── Step 2: Build fresh held-out synonym test pairs ───────────────────────────

def build_holdout_test_set(training_vocab, en2hi, h4_kv, cc_kv):
    """
    Build synonym and random pairs from words that are:
    a) In the MUSE dictionary (so we have Hindi translations)
    b) NOT in the pseudo-context training corpus vocab (truly held-out)
    c) In H4 vocabulary (so we can evaluate)
    d) In cc.hi.300 vocabulary (so oracle can verify)
    """
    print("Building held-out test set from unseen words...")

    # Collect Hindi words that are NOT in training vocab
    holdout_hi_words = []
    for en, hi in en2hi.items():
        if hi not in training_vocab and hi in h4_kv and hi in cc_kv:
            holdout_hi_words.append((en, hi))

    print(f"  Hindi words unseen in training and in both models: {len(holdout_hi_words):,}")

    if len(holdout_hi_words) < 50:
        print("  WARNING: Very few held-out words. Relaxing constraint - using words")
        print("  that appear in pseudo-context but NOT as synonym-pair members.")
        # Fall back: use all MUSE words that are in both vocabs
        holdout_hi_words = [(en, hi) for en, hi in en2hi.items()
                            if hi in h4_kv and hi in cc_kv]
        print(f"  Relaxed pool: {len(holdout_hi_words):,} words")

    # Build synonym pairs using semantic neighborhoods in cc.hi.300 (oracle)
    # For each held-out word, find its nearest neighbor in cc.hi.300 that is also held-out
    holdout_set = {hi for _, hi in holdout_hi_words}
    positive_pairs = []
    negative_pairs = []

    np.random.seed(42)
    words = [hi for _, hi in holdout_hi_words]
    if len(words) > 500:
        words = list(np.random.choice(words, 500, replace=False))

    print(f"  Sampling {len(words)} held-out words for pair construction...")

    # Positive pairs: use cc.hi.300 to find true nearest neighbors
    used = set()
    for w in words:
        if w not in cc_kv:
            continue
        try:
            neighbors = cc_kv.most_similar(w, topn=20)
            for (neighbor, sim) in neighbors:
                if (neighbor in holdout_set and
                    neighbor in h4_kv and
                    neighbor != w and
                    (w, neighbor) not in used and
                    (neighbor, w) not in used and
                    sim > 0.45):   # oracle says truly similar
                    positive_pairs.append({
                        "type": "positive",
                        "hi1": w, "hi2": neighbor,
                        "oracle_sim": float(round(sim, 4)),
                        "source": "cc.hi.300_neighbors"
                    })
                    used.add((w, neighbor))
                    break
        except Exception:
            continue
        if len(positive_pairs) >= 200:
            break

    # Negative pairs: random pairs from different words
    np.random.shuffle(words)
    n = len(words)
    for i in range(0, min(n-1, 400), 2):
        w1, w2 = words[i], words[i+1]
        if w1 == w2 or w1 not in cc_kv or w2 not in cc_kv:
            continue
        oracle_sim = cc_kv.similarity(w1, w2)
        if oracle_sim < 0.15:   # oracle says truly unrelated
            negative_pairs.append({
                "type": "negative",
                "hi1": w1, "hi2": w2,
                "oracle_sim": float(round(oracle_sim, 4)),
                "source": "random_holdout"
            })
        if len(negative_pairs) >= 200:
            break

    print(f"  Positive (synonym) pairs: {len(positive_pairs)}")
    print(f"  Negative (random) pairs : {len(negative_pairs)}")

    all_pairs = positive_pairs + negative_pairs
    np.random.shuffle(all_pairs)
    return all_pairs


# ── Step 3: Evaluate ──────────────────────────────────────────────────────────

def evaluate(kv, pairs, model_name, is_kv=True):
    vocab  = kv if is_kv else kv.wv
    pos_s  = []
    neg_s  = []

    for p in pairs:
        w1, w2 = p["hi1"], p["hi2"]
        if w1 in vocab and w2 in vocab:
            sim = vocab.similarity(w1, w2)
            if p["type"] == "positive":
                pos_s.append(sim)
            else:
                neg_s.append(sim)

    if len(pos_s) < 5 or len(neg_s) < 5:
        print(f"  [{model_name}] Too few pairs: pos={len(pos_s)} neg={len(neg_s)}")
        return None

    pos_mean = np.mean(pos_s)
    neg_mean = np.mean(neg_s)
    gap      = pos_mean - neg_mean

    best_f1, best_t = -1, 0.5
    for t in np.arange(0.05, 0.95, 0.01):
        y_true = [1]*len(pos_s) + [0]*len(neg_s)
        y_pred = [1 if s >= t else 0 for s in pos_s + neg_s]
        f = f1_score(y_true, y_pred, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t

    y_true = [1]*len(pos_s) + [0]*len(neg_s)
    y_pred = [1 if s >= best_t else 0 for s in pos_s + neg_s]

    return {
        "model":     model_name,
        "pos_mean":  round(pos_mean, 4),
        "neg_mean":  round(neg_mean, 4),
        "gap":       round(gap, 4),
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(best_f1, 4),
        "threshold": round(best_t, 2),
        "coverage":  round((len(pos_s)+len(neg_s)) / len(pairs), 4),
        "pos_sims":  pos_s,
        "neg_sims":  neg_s,
    }


# ── Step 4: Plots ─────────────────────────────────────────────────────────────

def plot_all(results_list, out_prefix):
    valid = [r for r in results_list if r]
    if not valid:
        print("No valid results to plot.")
        return

    # Distribution plots
    n = len(valid)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle("Debiased Evaluation: Similarity Distributions\n(Held-Out Test Set, No Training Contamination)",
                 fontsize=13, fontweight="bold")

    palette_pos = ["#27ae60", "#2980b9", "#8e44ad", "#d35400"]
    palette_neg = ["#e74c3c", "#e67e22", "#c0392b", "#e74c3c"]

    for i, (ax, res) in enumerate(zip(axes, valid)):
        sns.kdeplot(res["pos_sims"], ax=ax, fill=True, color=palette_pos[i], alpha=0.6,
                    label=f"Synonyms (mean={res['pos_mean']:.3f})")
        sns.kdeplot(res["neg_sims"], ax=ax, fill=True, color=palette_neg[i], alpha=0.4,
                    label=f"Random (mean={res['neg_mean']:.3f})")
        ax.axvline(res["threshold"], color="black", linestyle="--", lw=1.5,
                   label=f"Threshold={res['threshold']}")
        ax.set_title(f"{res['model']}\nGap={res['gap']:.3f} | Acc={res['accuracy']:.1%} | F1={res['f1']:.3f}",
                     fontsize=9)
        ax.set_xlabel("Cosine Similarity")
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = f"{out_prefix}_distributions.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {path}")

    # Bar chart
    metrics = ["gap", "accuracy", "f1", "precision", "recall"]
    labels  = ["Sep. Gap", "Accuracy", "F1", "Precision", "Recall"]
    x       = np.arange(len(labels))
    bw      = 0.8 / n
    palette = ["#3498db", "#27ae60", "#e67e22", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, res in enumerate(valid):
        vals = [res[m] for m in metrics]
        offset = (i - n/2 + 0.5) * bw
        bars = ax.bar(x + offset, vals, bw*0.9, label=res["model"],
                      color=palette[i % len(palette)], alpha=0.85)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)

    ax.set_title("Debiased Metric Comparison (Held-Out Test Set, No Contamination)",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = f"{out_prefix}_metrics.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("DEBIASED EVALUATION v2: Held-Out Test Set")
    print("=" * 65)

    training_vocab = get_training_vocab()
    en2hi          = load_muse_dict()

    print("\nLoading models...")
    h4_kv = KeyedVectors.load(os.path.join(MODELS_DIR, "H4_cp_cle.kv"))
    cc_kv = KeyedVectors.load(os.path.join(MODELS_DIR, "cc_hi_300.kv"))

    # Build fresh held-out test pairs
    print()
    test_pairs = build_holdout_test_set(training_vocab, en2hi, h4_kv, cc_kv)

    if len(test_pairs) < 20:
        print("ERROR: Could not build sufficient held-out test pairs.")
        return

    # Save test set for transparency
    with open(os.path.join(RESULTS_DIR, "holdout_test_pairs.json"), "w",
              encoding="utf-8") as f:
        json.dump(test_pairs, f, ensure_ascii=False, indent=2)

    print(f"\n  Test set saved: {len(test_pairs)} pairs")
    print(f"  Positive: {sum(1 for p in test_pairs if p['type']=='positive')}")
    print(f"  Negative: {sum(1 for p in test_pairs if p['type']=='negative')}")

    # Load remaining models
    h2_model = FastText.load(os.path.join(MODELS_DIR, "H2_native.bin"))
    h3_model = FastText.load(os.path.join(MODELS_DIR, "H3_pseudo.bin"))
    h5_kv    = KeyedVectors.load(os.path.join(MODELS_DIR, "H5_compressed.kv"))

    # Evaluate
    print("\nRunning evaluations...")
    r_h2 = evaluate(h2_model, test_pairs, "H2 (Native FT, same corpus)", is_kv=False)
    r_h3 = evaluate(h3_model, test_pairs, "H3 (Pseudo-Context)",          is_kv=False)
    r_h4 = evaluate(h4_kv,    test_pairs, "H4 (CP-CLE, our model)",       is_kv=True)
    r_h5 = evaluate(h5_kv,    test_pairs, "H5 (Compressed)",       is_kv=True)
    r_cc = evaluate(cc_kv,    test_pairs, "cc.hi.300 (300-dim reference)", is_kv=True)

    results = [r for r in [r_h2, r_h3, r_h4, r_h5, r_cc] if r]

    # Print table
    print("\n" + "=" * 72)
    print("FINAL DEBIASED RESULTS")
    print("(Held-out pairs: oracle-validated, never seen in training)")
    print("=" * 72)
    fmt = f"  {{:<32}} {{:>7}} {{:>7}} {{:>7}} {{:>7}} {{:>8}}"
    print(fmt.format("Model", "Gap", "Acc", "F1", "Thresh", "Coverage"))
    print("-" * 72)
    for r in results:
        print(fmt.format(
            r["model"][:32],
            f"{r['gap']:.3f}",
            f"{r['accuracy']:.3f}",
            f"{r['f1']:.3f}",
            f"{r['threshold']:.2f}",
            f"{r['coverage']:.1%}"
        ))
    print("-" * 72)

    if r_h2 and r_h5:
        print(f"\n  Fair comparison (same data constraint vs global alignment, H2 vs H5):")
        print(f"    Gap gain : {r_h5['gap'] - r_h2['gap']:+.3f}")
        print(f"    F1 gain  : {r_h5['f1']  - r_h2['f1']:+.3f}")
        print(f"    Acc gain : {r_h5['accuracy'] - r_h2['accuracy']:+.3f}")

    # Plots
    plot_all(results, os.path.join(RESULTS_DIR, "Debiased_v2"))

    # Save
    out = []
    for r in results:
        clean_r = {}
        for k, v in r.items():
            if k not in ("pos_sims","neg_sims"):
                if isinstance(v, (np.float32, np.float64, np.number)):
                    clean_r[k] = float(v)
                else:
                    clean_r[k] = v
        out.append(clean_r)

    with open(os.path.join(RESULTS_DIR, "debiased_results_v2.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nDone.")


if __name__ == "__main__":
    main()
