import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import FastText, KeyedVectors
import gensim.downloader as api

DATA_DIR = "data"
MODELS_DIR = "models"

def load_eval_vocab():
    out_path = os.path.join(DATA_DIR, "eval_pairs.json")
    with open(out_path, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
        
    en_vocab = set()
    hi_vocab = set()
    
    for item in eval_data:
        if item["type"] in ["positive", "negative"]:
            en_vocab.add(item["en1"])
            en_vocab.add(item["en2"])
            hi_vocab.add(item["hi1"])
            hi_vocab.add(item["hi2"])
            
    return list(en_vocab), list(hi_vocab)

def build_cache(en_vocab, glove_model):
    print("Building English similarity cache...")
    valid_en_words = [w for w in en_vocab if w in glove_model]
    
    # Pre-compute all pairwise similarities
    # To save memory and computation, we'll extract the vectors
    vectors = []
    word2idx = {}
    for i, w in enumerate(valid_en_words):
        vectors.append(glove_model[w])
        word2idx[w] = i
        
    E_tensor = torch.tensor(vectors, dtype=torch.float32)
    # Normalize to compute cosine similarity easily via dot product
    E_tensor = F.normalize(E_tensor, p=2, dim=1)
    
    # sim_E[i, j]
    sim_E = torch.mm(E_tensor, E_tensor.t())
    return valid_en_words, word2idx, sim_E

def get_bilingual_dictionary():
    out_path = os.path.join(DATA_DIR, "en-hi.txt")
    dictionary = {}
    with open(out_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                dictionary[parts[0]] = parts[1]
    return dictionary

def optimize_embeddings():
    print("Loading English pretrained embeddings (glove-wiki-gigaword-100)...")
    glove = api.load("glove-wiki-gigaword-100")
    
    en_vocab, hi_vocab = load_eval_vocab()
    dictionary = get_bilingual_dictionary()
    
    valid_en_words, word2idx, sim_E = build_cache(en_vocab, glove)
    
    print("Loading H3 (Pseudo-context) FastText model...")
    h3_path = os.path.join(MODELS_DIR, "H3_pseudo.bin")
    h3_model = FastText.load(h3_path)
    
    # Create the trainable Hindi embedding matrix
    # We only train the Hindi words that correspond to our valid_en_words
    hi_words = []
    en_words_mapped = []
    
    for en_w in valid_en_words:
        hi_w = dictionary.get(en_w)
        if hi_w and hi_w in h3_model.wv:
            hi_words.append(hi_w)
            en_words_mapped.append(en_w)
            
    print(f"Total valid mapped pairs for optimization: {len(hi_words)}")
    
    # Extract initial vectors
    initial_vectors = [h3_model.wv[hw] for hw in hi_words]
    H_tensor = torch.tensor(initial_vectors, dtype=torch.float32)
    
    # Define PyTorch parameter
    H = nn.Parameter(H_tensor)
    # Use vanilla SGD as instructed: H = H - lr * gradient
    optimizer = torch.optim.SGD([H], lr=0.5)
    
    indices_E = torch.tensor([word2idx[en_w] for en_w in en_words_mapped], dtype=torch.long)
    
    epochs = 10
    batch_size = 256
    num_pairs = 50000 # Sample 50000 random pairs per epoch
    
    print("Starting CP-CLE Optimization (Vanilla SGD over sampled pairs)...")
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = num_pairs // batch_size
        
        for _ in range(num_batches):
            optimizer.zero_grad()
            
            # Sample random pairs of indices (i, j)
            idx_i = torch.randint(0, len(hi_words), (batch_size,))
            idx_j = torch.randint(0, len(hi_words), (batch_size,))
            
            # Extract H vectors
            h_i = H[idx_i]
            h_j = H[idx_j]
            
            # Normalize to compute cosine similarity
            h_i_norm = F.normalize(h_i, p=2, dim=1)
            h_j_norm = F.normalize(h_j, p=2, dim=1)
            
            # sim_H for the sampled pairs
            sim_H = torch.sum(h_i_norm * h_j_norm, dim=1)
            
            # Extract target sim from sim_E
            e_i = indices_E[idx_i]
            e_j = indices_E[idx_j]
            target_sim = sim_E[e_i, e_j]
            
            # L_corr = (sim_E - sim_H)^2
            loss = F.mse_loss(sim_H, target_sim)
            loss.backward()
            
            # H = H - lr * gradient
            optimizer.step()
            
            # Normalize vectors after each update step
            with torch.no_grad():
                H.copy_(F.normalize(H, p=2, dim=1))
                
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} - Avg L_corr Loss: {(epoch_loss/num_batches):.6f}")
        
    print("Optimization finished.")
    
    # Save back to H4
    print("Building H4 KeyedVectors...")
    h4_kv = KeyedVectors(vector_size=h3_model.vector_size)
    
    # Copy all original H3 vectors first
    keys = h3_model.wv.index_to_key
    weights = h3_model.wv.vectors
    h4_kv.add_vectors(keys, weights)
    
    # Overwrite the updated vectors (normalized)
    with torch.no_grad():
        H_final = F.normalize(H, p=2, dim=1)
    updated_vectors = H_final.numpy()
    for idx, hw in enumerate(hi_words):
        h4_kv[hw] = updated_vectors[idx]
        
    out_path = os.path.join(MODELS_DIR, "H4_cp_cle.kv")
    h4_kv.save(out_path)
    print(f"H4 embeddings saved to {out_path}.")

if __name__ == "__main__":
    optimize_embeddings()
