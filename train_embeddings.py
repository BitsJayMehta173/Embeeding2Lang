import os
from gensim.models import FastText

DATA_DIR = "data"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

class CorpusIterator:
    def __init__(self, filepath):
        self.filepath = filepath

    def __iter__(self):
        with open(self.filepath, 'r', encoding='utf-8') as f:
            for line in f:
                yield line.strip().split()

def train_and_save_model(corpus_path, model_name):
    print(f"Training FastText model: {model_name} on {corpus_path}...")
    sentences = CorpusIterator(corpus_path)
    
    # Parameters from implementation plan: vector_size=100, window=5
    model = FastText(vector_size=100, window=5, min_count=1, workers=4)
    model.build_vocab(corpus_iterable=sentences)
    model.train(corpus_iterable=sentences, total_examples=model.corpus_count, epochs=5)
    
    out_path = os.path.join(MODELS_DIR, f"{model_name}.bin")
    model.save(out_path)
    print(f"Model {model_name} saved to {out_path}.")

if __name__ == "__main__":
    # H1: Translated Corpus
    train_and_save_model(os.path.join(DATA_DIR, "translated_corpus.txt"), "H1_translated")
    
    # H2: Native Hindi Corpus
    train_and_save_model(os.path.join(DATA_DIR, "hi_corpus.txt"), "H2_native")
    
    # H3: Pseudo-context Corpus
    train_and_save_model(os.path.join(DATA_DIR, "pseudo_context_corpus.txt"), "H3_pseudo")
    
    print("Embedding training completed.")
