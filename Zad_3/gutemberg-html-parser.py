from bs4 import BeautifulSoup
import re
import os

def zapisz_tekst_do_pliku(tekst, nazwa_pliku_docelowego):
    """
    Zapisuje tekst do określonego pliku tekstowego.
    """
    try:
        with open(nazwa_pliku_docelowego, 'w', encoding='utf-8') as plik:
            plik.write(tekst)
        print(f"\n✅ Sukces: Wyczyszczony tekst został zapisany do pliku '{nazwa_pliku_docelowego}'.")
        print(f"Lokalizacja pliku: {os.path.abspath(nazwa_pliku_docelowego)}")
    except Exception as e:
        print(f"Błąd podczas zapisu do pliku: {e}")

def wyluskaj_tekst_z_html(nazwa_pliku):
    """
    Wczytuje plik HTML i zwraca cały tekst, usuwając tagi HTML, 
    skrypty, style i inne zbędne elementy.

    :param nazwa_pliku: Ścieżka do pliku HTML.
    :return: Wyczyszczony tekst jako pojedynczy ciąg znaków.
    """
    try:
        # 1. Wczytanie zawartości pliku
        with open(nazwa_pliku, 'r', encoding='utf-8') as plik:
            zawartosc_html = plik.read()
        
        # 2. Utworzenie obiektu Beautiful Soup
        # 'html.parser' jest standardowym parserem Pythona
        soup = BeautifulSoup(zawartosc_html, 'html.parser')

        # 3. Usunięcie zbędnych elementów: skryptów i styli
        # Te elementy nie zawierają treści przeznaczonej dla użytkownika
        for element in soup(['script', 'style', 'head', 'title', 'meta', 'link']):
            element.decompose() # Usunięcie elementu z drzewa parsowania
        
        # 4. Wyłuskanie całego tekstu
        # .get_text() wyciąga zawartość tekstową ze wszystkich elementów
        tekst = soup.get_text()

        # 5. Opcjonalne czyszczenie i formatowanie
        # Zastąpienie wielu pustych linii/spacji pojedynczym znakiem nowej linii i usunięcie białych znaków na początku/końcu
        import re
        tekst_czysty = re.sub(r'(\s*\n\s*){2,}', '\n\n', tekst.strip())
        
        return tekst_czysty

    except FileNotFoundError:
        return f"Błąd: Plik '{nazwa_pliku}' nie został znaleziony."
    except Exception as e:
        return f"Wystąpił błąd podczas przetwarzania pliku: {e}"

# --- Przykład użycia ---
if __name__ == '__main__':  

    # 2. Wywołanie funkcji
    tekst_wyczyszczony = wyluskaj_tekst_z_html('The Pickwick Papers, by Charles Dickens.html')

    zapisz_tekst_do_pliku(tekst_wyczyszczony, 'wyczyszczony_tekst.txt')