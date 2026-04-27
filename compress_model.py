"""
compress_model.py
=================
Compresses the high-quality cc.hi.300 model into an ultra-lightweight
model suitable for Edge AI, without using destructive localized alignment.

Steps:
1. Truncate vocabulary to the top 20,000 most frequent words.
2. Use Principal Component Analysis (PCA) to reduce dimensionality
   from 300 to 100, preserving global semantic variance.
   
This results in an ~8 MB model that retains the global structure of
the 1.3 GB Facebook model, solving both our generalization and size issues.
"""

import os
import numpy as np
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

TOP_K = 20000
TARGET_DIM = 100

def main():
    print(f"Loading cc.hi.300 (this may take a minute)...")
    cc_kv = KeyedVectors.load(os.path.join(MODELS_DIR, "cc_hi_300.kv"))
    
    print(f"\n1. Truncating to top {TOP_K} words...")
    words = cc_kv.index_to_key[:TOP_K]
    vectors = np.array([cc_kv[w] for w in words])
    
    print(f"2. Reducing dimensions from 300 to {TARGET_DIM} using PCA...")
    pca = PCA(n_components=TARGET_DIM)
    reduced_vectors = pca.fit_transform(vectors)
    
    # Re-normalize to unit length
    reduced_vectors = reduced_vectors / np.linalg.norm(reduced_vectors, axis=1, keepdims=True)
    
    var_preserved = sum(pca.explained_variance_ratio_)
    print(f"   Variance preserved: {var_preserved:.1%}")
    
    print(f"\n3. Saving lightweight model...")
    h5_kv = KeyedVectors(vector_size=TARGET_DIM)
    h5_kv.add_vectors(words, reduced_vectors)
    
    output_path = os.path.join(MODELS_DIR, "H5_compressed.kv")
    h5_kv.save(output_path)
    
    # Calculate actual size
    file_size_mb = (len(words) * TARGET_DIM * 4) / (1024 * 1024)
    
    print("\n" + "="*50)
    print("COMPRESSION COMPLETE")
    print("="*50)
    print(f"Model:      H5 (Compressed cc.hi.300)")
    print(f"Vocab:      {TOP_K} words")
    print(f"Dimensions: {TARGET_DIM}")
    print(f"Est. Size:  {file_size_mb:.1f} MB")
    print("="*50)

if __name__ == "__main__":
    main()
