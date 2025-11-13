import numpy as np
import json
import logging
from gensim.models import Word2Vec
from tokenizers import Tokenizer
import os

# Ustawienie logowania (na wszelki wypadek)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# --- KONFIGURACJA ŚCIEŻEK ---
# Plik modelu (musi być tym, który został zapisany przez cbow_trainer.py)
INPUT_MODEL_FILE = "./output/embedding_word2vec_cbow_model.model"
# Plik tokenizera (potrzebny do tokenizacji słów testowych)
TOKENIZER_FILE = "./tokenizers/all-tokenizer.json"


def get_word_vector_and_similar(word: str, tokenizer: Tokenizer, model: Word2Vec, topn: int = 10):
    """
    Tokenizuje słowo na tokeny podwyrazowe, uśrednia wektory tokenów,
    a następnie szuka najbardziej podobnych tokenów do tego uśrednionego wektora.
    """
    # 1. Tokenizacja słowa
    # Dodanie spacji na początku może pomóc w poprawnym tokenizowaniu
    # tokenizatorami BPE/SentencePiece (tokeny zaczynają się od '_')
    encoding = tokenizer.encode(" " + word + " ") 
    word_tokens = [t.strip() for t in encoding.tokens if t.strip()]
    
    # Prosta próba usunięcia tokenów początku/końca (zależne od konfiguracji tokenizera)
    if word_tokens and word_tokens[0] in ['[CLS]', '<s>', '<s>', 'Ġ']:
        word_tokens = word_tokens[1:]
    if word_tokens and word_tokens[-1] in ['[SEP]', '</s>', '</s>']:
        word_tokens = word_tokens[:-1]
        
    valid_vectors = []
    
    # 2. Zbieranie wektorów dla każdego tokenu
    for token in word_tokens:
        # Sprawdzamy, czy token jest w słowniku modelu Word2Vec
        if token in model.wv:
            valid_vectors.append(model.wv[token])

    if not valid_vectors:
        print(f"BŁĄD: Słowo '{word}' (tokeny: {word_tokens}) nie zawiera tokenów, które są w słowniku modelu.")
        return None, None, word_tokens

    # 3. Uśrednianie wektorów (wektor dla całego słowa)
    word_vector = np.mean(valid_vectors, axis=0)

    # 4. Znalezienie najbardziej podobnych tokenów
    similar_tokens = model.wv.most_similar(
        positive=[word_vector],
        topn=topn
    )
    
    return word_vector, similar_tokens, word_tokens


def run_inference():
    try:
        # Wczytywanie modelu Word2Vec
        print(f"Ładowanie modelu Word2Vec z pliku: {INPUT_MODEL_FILE}")
        model = Word2Vec.load(INPUT_MODEL_FILE)
    except FileNotFoundError:
        print(f"BŁĄD: Nie znaleziono pliku modelu '{INPUT_MODEL_FILE}'. Uruchom najpierw cbow_trainer.py.")
        return
    
    # Wczytywanie tokenizera
    try:
        print(f"Ładowanie tokenizera z pliku: {TOKENIZER_FILE}")
        tokenizer = Tokenizer.from_file(TOKENIZER_FILE)
    except FileNotFoundError:
        print(f"BŁĄD: Nie znaleziono pliku tokenizera '{TOKENIZER_FILE}'.")
        return

    
    print("\n--- Weryfikacja: Szukanie podobieństw dla całych SŁÓW (uśrednianie wektorów tokenów) ---")

    # Słowa do przetestowania, jak w przykładzie zadania
    words_to_test = ['wojsko', 'szlachta', 'choroba', 'król'] 

    for word in words_to_test:
        word_vector, similar_tokens, token_list = get_word_vector_and_similar(word, tokenizer, model, topn=10)
        
        if word_vector is not None:
            print(f"\n10 tokenów najbardziej podobnych do SŁOWA '{word}' (tokeny: {token_list}):")
            # Wyświetlanie wektora (pierwsze 5 elementów)
            print(f"  > Wektor słowa (początek): {word_vector[:5]}...")
            
            for token, similarity in similar_tokens:
                print(f"  - {token}: {similarity:.4f}")

    # --- WERYFIKACJA DLA WZORCA MATEMATYCZNEGO (Analogia wektorowa) ---
    tokens_analogy = ['dziecko', 'kobieta']

    print(f"\n--- Weryfikacja: Analogia wektorowa dla tokenów: {tokens_analogy} ---")

    # Sprawdzamy czy tokeny są w słowniku
    if all(token in model.wv for token in tokens_analogy):
        similar_to_combined = model.wv.most_similar(
            positive=tokens_analogy,
            topn=10
        )

        print(f"10 tokenów najbardziej podobnych do kombinacji tokenów: {tokens_analogy}")
        for token, similarity in similar_to_combined:
            print(f"  - {token}: {similarity:.4f}")
    else:
        print(f"Ostrzeżenie: Co najmniej jeden z tokenów '{tokens_analogy}' nie znajduje się w słowniku. Pomięto analogię.")

if __name__ == "__main__":
    run_inference()