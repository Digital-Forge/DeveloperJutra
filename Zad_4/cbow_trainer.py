# cbow_trainer.py

import numpy as np
import json
import logging
from gensim.models import Word2Vec
from tokenizers import Tokenizer
import os
# import z corpora (zakładam, że jest to plik pomocniczy)
from corpora import CORPORA_FILES # type: ignore 

# Ustawienie logowania dla gensim
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# --- KONFIGURACJA ŚCIEŻEK I PARAMETRÓW ---
# Pliki korpusowe
files = CORPORA_FILES["ALL"]

# Ścieżka do tokenizera (pamiętaj o dostosowaniu ścieżek!)
TOKENIZER_FILE = "./tokenizers/all-tokenizer.json" 

# Ścieżki zapisu
OUTPUT_DIR = "./output"
OUTPUT_TENSOR_FILE = os.path.join(OUTPUT_DIR, "embedding_tensor_cbow.npy")
OUTPUT_MAP_FILE = os.path.join(OUTPUT_DIR, "embedding_token_to_index_map.json")
OUTPUT_MODEL_FILE = os.path.join(OUTPUT_DIR, "embedding_word2vec_cbow_model.model")

# Parametry treningu Word2Vec (CBOW)
VECTOR_LENGTH = 40
WINDOW_SIZE = 8
MIN_COUNT = 2
WORKERS = 10
EPOCHS = 40
SAMPLE_RATE = 1e-2
SG_MODE = 0


def aggregate_raw_sentences(files):
    """Wczytuje i agreguje surowe zdania z listy plików."""
    raw_sentences = []
    print("Wczytywanie tekstu z plików...")
    for file in files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
                raw_sentences.extend(lines)
        except FileNotFoundError:
            print(f"OSTRZEŻENIE: Nie znaleziono pliku '{file}'. Pomijam.")
            continue

    if not raw_sentences:
        raise ValueError("BŁĄD: Pliki wejściowe są puste lub nie zostały wczytane.")
    return raw_sentences

def train_cbow_model():
    # Upewnienie się, że katalog wyjściowy istnieje
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Krok 1: Ładowanie tokenizera
    print(f"Ładowanie tokenizera z pliku: {TOKENIZER_FILE}")
    try:
        tokenizer = Tokenizer.from_file(TOKENIZER_FILE)
    except FileNotFoundError:
        print(f"BŁĄD: Nie znaleziono pliku '{TOKENIZER_FILE}'. Upewnij się, że plik istnieje.")
        return
    
    # Krok 2: Wczytywanie i tokenizacja danych
    try:
        raw_sentences = aggregate_raw_sentences(files)
    except ValueError as e:
        print(e)
        return

    print(f"Tokenizacja {len(raw_sentences)} zdań...")
    encodings = tokenizer.encode_batch(raw_sentences)
    tokenized_sentences = [encoding.tokens for encoding in encodings]
    print(f"Przygotowano {len(tokenized_sentences)} sekwencji do treningu.")

    # Krok 3: Trening Word2Vec (CBOW)
    print("\n--- Rozpoczynanie Treningu Word2Vec (CBOW) ---")
    model = Word2Vec(
        sentences=tokenized_sentences,
        vector_size=VECTOR_LENGTH,
        window=WINDOW_SIZE,
        min_count=MIN_COUNT,
        workers=WORKERS,
        sg=SG_MODE, # 0: CBOW
        epochs=EPOCHS,
        sample=SAMPLE_RATE,
    )
    print("Trening zakończony pomyślnie.")

    # Krok 4: Eksport i Zapis Wyników
    
    # Zapisanie pełnego modelu gensim (do użycia w wnioskowaniu)
    model.save(OUTPUT_MODEL_FILE)
    print(f"Pełny model Word2Vec zapisany jako: '{OUTPUT_MODEL_FILE}'.")

    # Opcjonalnie: Zapisanie tensora i mapowania dla innych narzędzi (np. TensorBoard)
    embedding_matrix_tensor = np.array(model.wv.vectors, dtype=np.float32)
    np.save(OUTPUT_TENSOR_FILE, embedding_matrix_tensor)
    print(f"Tensor embeddingowy zapisany jako: '{OUTPUT_TENSOR_FILE}'.")
    
    token_to_index = {token: model.wv.get_index(token) for token in model.wv.index_to_key}
    with open(OUTPUT_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(token_to_index, f, ensure_ascii=False, indent=4)
    print(f"Mapa tokenów do indeksów zapisana jako: '{OUTPUT_MAP_FILE}'.")


if __name__ == "__main__":
    train_cbow_model()