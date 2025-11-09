from tokenizers import Tokenizer
from corpora import get_corpus_file
import os
import glob
import sys

# Definicja dostępnych tokenizerów (klucz: nazwa, wartość: ścieżka do pliku JSON)
# Automatyczne wykrywanie plików w podkatalogu 'tokenizers'
TOKENIZERS = {}
tokenizer_files = glob.glob('tokenizers/*.json')

if not tokenizer_files:
    print("Ostrzeżenie: Nie znaleziono żadnych plików *.json w katalogu 'tokenizers/'.")

for path in tokenizer_files:
    # Wydobycie nazwy tokenizera z nazwy pliku
    base_name = os.path.basename(path)
    name_without_ext = os.path.splitext(base_name)[0]
    TOKENIZERS[name_without_ext] = path

# KRYTYCZNY BŁĄD: Sprawdzenie, czy w ogóle mamy jakieś tokenizery do pracy
if not TOKENIZERS:
    print("Krytyczny błąd: Brak dostępnych tokenizerów. Zamykanie programu.")
    sys.exit(1)

# --- POBIERANIE DANYCH WEJŚCIOWYCH OD UŻYTKOWNIKA ---
# Prośba o podanie DOKŁADNEGO wzorca tekstu
text_input = input("Podaj nazwę : ")
corpus_pattern = text_input.strip()
text_name = corpus_pattern.replace("*", "").replace(".txt", "")

if not corpus_pattern:
    print("Krytyczny błąd: Wymagana nazwa tekstu nie została podana. Zamykanie programu.")
    sys.exit(1)

print(f"Szukanie plików")

# Wyszukiwanie plików korpusu
corpus_files = get_corpus_file("WOLNELEKTURY", corpus_pattern)
source_txt = ""

if not corpus_files:
    # Jeśli pliki nie zostaną znalezione, to błąd krytyczny
    print(f"Krytyczny błąd: Nie znaleziono plików korpusu dla wzorca '{corpus_pattern}'. Zamykanie programu.")
    sys.exit(1)
else:
    # Pliki znalezione, próbujemy wczytać pierwszy
    try:
        source_path = corpus_files[0]
        print(f"Znaleziono {len(corpus_files)} plików. Wczytuję pierwszy: {source_path}")
        with open(source_path, 'r', encoding='utf-8') as f:
            source_txt = f.read()
        
        if not source_txt.strip():
            # Sprawdzenie, czy plik jest pusty (puste jest błędem)
            print(f"Krytyczny błąd: Plik '{source_path}' jest pusty. Zamykanie programu.")
            sys.exit(1)
            
        print(f"Wczytano {len(source_txt)} znaków.")
        
    except Exception as e:
        # Jeśli wczytywanie zawiedzie, to błąd krytyczny
        print(f"Krytyczny błąd: Błąd podczas wczytywania pliku '{source_path}': {e}. Zamykanie programu.")
        sys.exit(1)

# Lista do przechowywania wyników dla każdego tokenizera
all_results = []

# 2. Przetwarzanie tekstu przez każdy znaleziony tokenizer
print("\n--- Rozpoczynanie tokenizacji dla wszystkich znalezionych modeli ---")

for name, path in TOKENIZERS.items():
    print(f"Przetwarzam tokenizer: {name}")
    
    try:
        # Ładowanie tokenizera
        tokenizer = Tokenizer.from_file(path)
        
        # Kodowanie tekstu
        encoded = tokenizer.encode(source_txt)
        token_count = len(encoded.ids)
        
        # Zapisywanie wyniku do listy
        result_line = f"Tokenizer: {name}, Liczba tokenów: {token_count}"
        all_results.append(result_line)
        print(f"   -> Liczba tokenów: {token_count}")
        
    except FileNotFoundError:
        # Błędy pojedynczego tokenizera są zapisywane jako błędy, ale nie przerywają analizy
        error_line = f"Tokenizer: {name}, BŁĄD: Plik tokenizera nie został znaleziony ({path})"
        all_results.append(error_line)
        print(f"   -> BŁĄD: Plik nie znaleziony.")
    except Exception as e:
        error_line = f"Tokenizer: {name}, BŁĄD: Wystąpił nieznany błąd podczas kodowania: {e}"
        all_results.append(error_line)
        print(f"   -> BŁĄD kodowania: {e}")

# 3. Zapisywanie wszystkich wyników do pojedynczego pliku
file_name = f"logs/tokenized-{text_name}-wyniki.txt"

output_content = (
    f"--- Wyniki tokenizacji dla tekstu '{text_name}' (Wzorzec: {corpus_pattern}) ---\n\n" +
    "\n".join(all_results) +
    f"\n\nAnaliza ukończona dla {len(TOKENIZERS)} tokenizatorów."
)

try:
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(output_content)
    
    print(f"\nGotowe! Wyniki zapisano do pliku: {os.path.abspath(file_name)}")
except Exception as e:
    print(f"\nKrytyczny błąd: Błąd podczas zapisu pliku wynikowego: {e}. Zamykanie programu.")
    sys.exit(1)