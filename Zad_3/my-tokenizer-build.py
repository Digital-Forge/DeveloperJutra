import sys
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from corpora import get_corpus_file, CORPORA_FILES # Wprowadzono CORPORA_FILES, aby sprawdzić, czy korpus istnieje

def get_glob_pattern():
    """Pobiera wzorzec glob dla plików od użytkownika."""
    try:
        pattern = input("Podaj wzorzec plików (np. *.txt, NKJP, ALL): ").strip()
        if not pattern:
            return "*.txt"
        return pattern
    except EOFError:
        print("\nPrzerwanie przez użytkownika.")
        sys.exit(1)
    except Exception as e:
        print(f"Wystąpił nieoczekiwany błąd: {e}")
        sys.exit(1)


# Pobranie wzorca glob od użytkownika
GLOB_PATTERN = get_glob_pattern()
source_corpus_name = "WOLNELEKTURY"
FILES = []

pattern_upper = GLOB_PATTERN.upper()

if "ALL" in pattern_upper:
    # W oryginalnym corpora.py, "ALL" jest listą plików
    FILES = [str(f) for f in CORPORA_FILES["ALL"]]
    print(f"Wybrano korpus ALL. Znaleziono {len(FILES)} plików.")

elif "NKJP" in pattern_upper:
    source_corpus_name = "NKJP"
    FILES = [str(f) for f in get_corpus_file(source_corpus_name, "*.txt")]
    print(f"Wybrano korpus NKJP. Znaleziono {len(FILES)} plików dla wzorca '{GLOB_PATTERN}'.")

elif "WOLNELEKTURY" in pattern_upper:
    source_corpus_name = "WOLNELEKTURY"
    FILES = [str(f) for f in get_corpus_file(source_corpus_name, "*.txt")]
    print(f"Wybrano korpus WOLNELEKTURY. Znaleziono {len(FILES)} plików dla wzorca '{GLOB_PATTERN}'.")

else:
    FILES = [str(f) for f in get_corpus_file(source_corpus_name, GLOB_PATTERN)]
    print(f"Wybrano korpus WOLNELEKTURY. Znaleziono {len(FILES)} plików dla wzorca '{GLOB_PATTERN}'.")


if not FILES:
    print("Błąd: Nie znaleziono plików do trenowania! Sprawdź wzorzec w corpora.py.")
    sys.exit(1)

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    vocab_size=32000,
    min_frequency=2
)

print("Rozpoczęcie trenowania tokenizera...")
tokenizer.train(FILES, trainer=trainer)
print("Trenowanie zakończone.")

safe_pattern = GLOB_PATTERN.replace('*', '').replace('.txt', '')
TOKENIZER_OUTPUT_FILE = f"tokenizers/{safe_pattern.lower()}-tokenizer.json"
tokenizer.save(TOKENIZER_OUTPUT_FILE)
print(f"Tokenizacja zapisana w: {TOKENIZER_OUTPUT_FILE}")
