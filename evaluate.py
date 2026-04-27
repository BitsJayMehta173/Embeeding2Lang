import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import FastText
from nltk.corpus import wordnet

DATA_DIR = "data"
MODELS_DIR = "models"
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_dictionary():
    out_path = os.path.join(DATA_DIR, "en-hi.txt")
    dictionary = {}
    with open(out_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                en, hi = parts
                dictionary[en] = hi
    return dictionary

def build_test_sets(dictionary, num_pairs=1000):
    positive_pairs = []
    en_words = list(dictionary.keys())
    
    # Positive pairs: Hindi translations of English synonyms
    for en_word in en_words:
        if len(positive_pairs) >= num_pairs:
            break
        synonyms = set()
        for syn in wordnet.synsets(en_word):
            for l in syn.lemmas():
                synonyms.add(l.name().lower())
        
        if en_word in synonyms:
            synonyms.remove(en_word)
            
        hi_word1 = dictionary[en_word]
        for syn in synonyms:
            if syn in dictionary:
                hi_word2 = dictionary[syn]
                if hi_word1 != hi_word2:
                    positive_pairs.append((hi_word1, hi_word2))
                    break # just add one pair per word to diversify
    
    # Ensure we don't have more than needed
    positive_pairs = positive_pairs[:num_pairs]
    
    # Negative pairs: Random Hindi words
    hi_words = list(dictionary.values())
    negative_pairs = []
    for _ in range(len(positive_pairs)):
        w1 = random.choice(hi_words)
        w2 = random.choice(hi_words)
        negative_pairs.append((w1, w2))
        
    # Cross-lingual pairs: (en_word, hi_word)
    cross_lingual_pairs = []
    for en_word in en_words:
        if len(cross_lingual_pairs) >= num_pairs:
            break
        cross_lingual_pairs.append((en_word, dictionary[en_word]))
        
    return positive_pairs, negative_pairs, cross_lingual_pairs

def evaluate_model(model_path, pos_pairs, neg_pairs, cross_pairs):
    try:
        model = FastText.load(model_path)
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        return [], [], []

    def get_sims(pairs):
        sims = []
        for w1, w2 in pairs:
            if w1 in model.wv and w2 in model.wv:
                sims.append(model.wv.similarity(w1, w2))
        return sims

    pos_sims = get_sims(pos_pairs)
    neg_sims = get_sims(neg_pairs)
    cross_sims = get_sims(cross_pairs)
    
    return pos_sims, neg_sims, cross_sims

def plot_distributions(all_results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, (model_name, (pos, neg, cross)) in enumerate(all_results.items()):
        ax = axes[i]
        if pos:
            sns.kdeplot(pos, ax=ax, label="Positive (Synonyms)", fill=True)
        if neg:
            sns.kdeplot(neg, ax=ax, label="Negative (Random)", fill=True)
        if cross:
            sns.kdeplot(cross, ax=ax, label="Cross-lingual", fill=True)
            
        ax.set_title(f"{model_name} Similarity Distribution")
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Density")
        ax.legend()
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "similarity_distributions.png"))
    plt.close()

def main():
    print("Loading dictionary and building test sets...")
    dictionary = load_dictionary()
    pos_pairs, neg_pairs, cross_pairs = build_test_sets(dictionary, num_pairs=500)
    
    models = {
        "H1_Translated": os.path.join(MODELS_DIR, "H1_translated.bin"),
        "H2_Native": os.path.join(MODELS_DIR, "H2_native.bin"),
        "H3_Pseudo": os.path.join(MODELS_DIR, "H3_pseudo.bin")
    }
    
    all_results = {}
    summary_data = []
    
    print("Evaluating models...")
    for model_name, model_path in models.items():
        pos, neg, cross = evaluate_model(model_path, pos_pairs, neg_pairs, cross_pairs)
        all_results[model_name] = (pos, neg, cross)
        
        pos_mean = np.mean(pos) if pos else 0
        neg_mean = np.mean(neg) if neg else 0
        cross_mean = np.mean(cross) if cross else 0
        
        summary_data.append({
            "Model": model_name,
            "Positive_Mean": pos_mean,
            "Negative_Mean": neg_mean,
            "CrossLingual_Mean": cross_mean,
            "Pairs_Evaluated": f"P:{len(pos)} N:{len(neg)} C:{len(cross)}"
        })
        
    plot_distributions(all_results)
    
    df = pd.DataFrame(summary_data)
    df.to_csv(os.path.join(OUTPUT_DIR, "evaluation_summary.csv"), index=False)
    
    print("\n--- Evaluation Summary ---")
    print(df.to_string(index=False))
    print("\nPlots and summary saved to 'results' directory.")

if __name__ == "__main__":
    main()
