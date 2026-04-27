import os
import re
import urllib.request
import nltk
from datasets import load_dataset
import pandas as pd
import json

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def download_muse_dict():
    url = "https://dl.fbaipublicfiles.com/arrival/dictionaries/en-hi.txt"
    out_path = os.path.join(DATA_DIR, "en-hi.txt")
    if not os.path.exists(out_path):
        print("Downloading MUSE en-hi dictionary...")
        urllib.request.urlretrieve(url, out_path)
    print("MUSE dictionary downloaded.")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_and_save_wikipedia(lang, num_sentences=500000):
    out_path = os.path.join(DATA_DIR, f"{lang}_corpus.txt")
    if os.path.exists(out_path):
        print(f"Corpus for {lang} already exists.")
        return
    
    print(f"Downloading Wikipedia subset for {lang}...")
    dataset = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train", streaming=True)
    
    sentences_collected = []
    
    for item in dataset:
        text = item['text']
        sentences = nltk.sent_tokenize(text)
        for sent in sentences:
            cleaned = clean_text(sent)
            if len(cleaned.split()) >= 5: # Keep sentences with at least 5 words
                sentences_collected.append(cleaned)
                if len(sentences_collected) >= num_sentences:
                    break
        if len(sentences_collected) >= num_sentences:
            break
            
    with open(out_path, 'w', encoding='utf-8') as f:
        for sent in sentences_collected:
            f.write(sent + '\n')
            
    print(f"Saved {len(sentences_collected)} sentences to {out_path}.")

def load_muse_dict():
    out_path = os.path.join(DATA_DIR, "en-hi.txt")
    dictionary = {}
    with open(out_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                en, hi = parts
                dictionary[en] = hi
    return dictionary

if __name__ == "__main__":
    print("Starting data preparation...")
    download_muse_dict()
    process_and_save_wikipedia("en", num_sentences=500000)
    process_and_save_wikipedia("hi", num_sentences=500000)
    print("Data preparation completed.")
