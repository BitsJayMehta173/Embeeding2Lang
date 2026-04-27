import os
import json
import random
from nltk.corpus import wordnet

DATA_DIR = "data"

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

def build_test_sets(dictionary, num_pairs=500):
    eval_data = []
    en_words = list(dictionary.keys())
    
    # 1. Positive pairs: Hindi translations of English synonyms
    pos_count = 0
    for en_word in en_words:
        if pos_count >= num_pairs:
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
                    eval_data.append({
                        "type": "positive",
                        "en1": en_word,
                        "en2": syn,
                        "hi1": hi_word1,
                        "hi2": hi_word2
                    })
                    pos_count += 1
                    break 
                    
    # 2. Negative pairs: Random words
    # We pair random English words and their Hindi translations
    for _ in range(num_pairs):
        en1 = random.choice(en_words)
        en2 = random.choice(en_words)
        hi1 = dictionary[en1]
        hi2 = dictionary[en2]
        eval_data.append({
            "type": "negative",
            "en1": en1,
            "en2": en2,
            "hi1": hi1,
            "hi2": hi2
        })
        
    # 3. Cross-lingual pairs: (en_word, hi_word)
    # We store the same logic, but for cross-lingual evaluation we compare en1 and hi1
    for i in range(num_pairs):
        en_word = en_words[i]
        hi_word = dictionary[en_word]
        eval_data.append({
            "type": "cross_lingual",
            "en1": en_word,
            "en2": en_word, # placeholder
            "hi1": hi_word,
            "hi2": hi_word  # placeholder
        })
        
    return eval_data

if __name__ == "__main__":
    print("Building consistent evaluation set...")
    dictionary = load_dictionary()
    eval_data = build_test_sets(dictionary, num_pairs=500)
    
    out_path = os.path.join(DATA_DIR, "eval_pairs.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)
        
    print(f"Saved {len(eval_data)} evaluation pairs to {out_path}.")
