import os
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

def generate_translated_corpus(dictionary):
    print("Generating Translated Corpus (H1)...")
    en_corpus_path = os.path.join(DATA_DIR, "en_corpus.txt")
    out_path = os.path.join(DATA_DIR, "translated_corpus.txt")
    
    with open(en_corpus_path, 'r', encoding='utf-8') as f_in, \
         open(out_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            words = line.strip().split()
            translated_words = [dictionary.get(w, w) for w in words]
            f_out.write(" ".join(translated_words) + '\n')
    print(f"Translated Corpus saved to {out_path}.")

def generate_pseudo_context_corpus(dictionary):
    print("Generating Pseudo-context Corpus (H3)...")
    out_path = os.path.join(DATA_DIR, "pseudo_context_corpus.txt")
    
    # We will generate synthetic sentences for each English word in the dictionary
    with open(out_path, 'w', encoding='utf-8') as f_out:
        for en_word, hi_word in dictionary.items():
            synonyms = set()
            for syn in wordnet.synsets(en_word):
                for l in syn.lemmas():
                    synonyms.add(l.name().lower())
            
            # Remove the word itself
            if en_word in synonyms:
                synonyms.remove(en_word)
            
            hi_synonyms = [dictionary[s] for s in synonyms if s in dictionary]
            
            if hi_synonyms:
                # Create a synthetic sentence: hi_word hi_synonym1 hi_synonym2 ...
                # We can also add some randomness to the context by shuffling or repeating
                sentence = [hi_word] + hi_synonyms + [en_word]
                random.shuffle(sentence)
                f_out.write(" ".join(sentence) + '\n')
                
    print(f"Pseudo-context Corpus saved to {out_path}.")

if __name__ == "__main__":
    print("Loading dictionary...")
    dictionary = load_dictionary()
    
    generate_translated_corpus(dictionary)
    generate_pseudo_context_corpus(dictionary)
    print("Dataset variant generation completed. (Native Hindi corpus is already hi_corpus.txt)")
