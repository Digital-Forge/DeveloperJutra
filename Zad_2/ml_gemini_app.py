import mlflow
import google.genai as genai
from dotenv import load_dotenv
import os

# 1. Wczytanie klucza z pliku .env
load_dotenv() 

# 2. Ustawienie adresu serwera śledzenia (`mlflow ui`)
mlflow.set_tracking_uri("http://127.0.0.1:5000") 

# 3. Włączenie automatycznego śledzenia dla Gemini (Autologging)
mlflow.gemini.autolog()

mlflow.set_experiment("Gemini_Tracing") 

# 4. Rozpoczęcie Uruchomienia MLflow
with mlflow.start_run() as run:
    print(f"MLflow Run ID: {run.info.run_id}")

    try:
        # Klient automatycznie użyje klucza z GEMINI_API_KEY
        client = genai.Client()

        prompt = "Wytłumacz, czym jest uczenie maszynowe, używając analogii do gotowania."
        print(f"\n--- Zapytanie: {prompt} ---")

        # Wywołanie API Gemini
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=prompt
        )

        print("\n--- Odpowiedź ---")
        print(response.text)

    except Exception as e:
        print(f"Błąd: {e}")
        print("Sprawdź, czy klucz API jest poprawny.")

print(f"\nUruchomienie zakończone. Trasa komunikacji została zapisana.")