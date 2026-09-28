# Lokalny asystent do pytań o dokumenty PDF

Prosty program typu RAG (*Retrieval-Augmented Generation*), który wczytuje tekst z dokumentu PDF, wyszukuje fragmenty powiązane z pytaniem i przekazuje je lokalnemu modelowi językowemu. Odpowiedź jest generowana na podstawie znalezionego kontekstu.

## Wykorzystywane technologie

- **Ollama** – lokalne generowanie embeddingów i odpowiedzi.
- **ChromaDB** – wyszukiwanie fragmentów według podobieństwa wektorowego.
- **pypdf** – odczytywanie tekstu z plików PDF.

## Wymagania

- Python 3.9 lub nowszy
- Zainstalowana i uruchomiona aplikacja [Ollama](https://ollama.com/)
- Modele Ollama:
  - `nomic-embed-text` – tworzenie embeddingów,
  - `phi3` – generowanie odpowiedzi.

Pobierz wymagane modele:

```bash
ollama pull nomic-embed-text
ollama pull phi3
```

Zainstaluj biblioteki Pythona:

```bash
python3 -m pip install ollama chromadb pypdf
```

## Struktura projektu

```text
.
├── main.py
├── komunikacja_model.py
└── input/
    └── sprawozdanie_fizyka.pdf
```

## Uruchomienie

Umieść dokument PDF w katalogu `input/`, a następnie uruchom program:

```bash
python3 main.py
```

Domyślnie program przetwarza plik `input/sprawozdanie_fizyka.pdf` i zadaje pytanie:

> Jaki jest wzór na moc wiatru?

Ścieżkę do dokumentu oraz treść pytania można zmienić w pliku `main.py`.

## Jak działa program

### 1. Wczytanie dokumentu

Funkcja `extract_text_from_pdf()` w pliku `komunikacja_model.py` odczytuje tekst ze wszystkich stron PDF-a za pomocą biblioteki `pypdf`. Strony bez możliwego do wyodrębnienia tekstu są pomijane.

### 2. Podział tekstu na fragmenty

Wyodrębniony tekst jest dzielony na fragmenty po 800 znaków. Kolejny fragment zaczyna się 700 znaków po początku poprzedniego, więc sąsiadujące fragmenty nakładają się na siebie o 100 znaków. Nakładanie pomaga zachować kontekst na granicach fragmentów.

### 3. Indeksowanie w ChromaDB

Funkcja `setup_vector_db()` tworzy kolekcję `dokumenty_rag`. Dla każdego fragmentu Ollama, używając modelu `nomic-embed-text`, generuje embedding. Fragmenty, ich embeddingi i identyfikatory są następnie dodawane do kolekcji ChromaDB.

### 4. Wyszukanie kontekstu

Funkcja `query_rag()` tworzy embedding pytania przy użyciu tego samego modelu. ChromaDB wyszukuje trzy najbardziej pasujące fragmenty — domyślna wartość parametru `n_results` wynosi `3`.

### 5. Wygenerowanie odpowiedzi

Znalezione fragmenty są łączone w kontekst i dołączane do promptu. Model `phi3` otrzymuje polecenie, aby odpowiedzieć na pytanie na podstawie tego kontekstu, a w przypadku braku informacji wprost przyznać, że jej nie zna. Odpowiedź jest wypisywana strumieniowo w terminalu.

## Ważne informacje

- W obecnej wersji ChromaDB jest tworzona przez `chromadb.Client()`, więc baza działa w pamięci i nie jest trwale zapisywana między uruchomieniami programu.
- Dokument PDF musi zawierać tekst możliwy do wyodrębnienia. Skanowane dokumenty mogą wymagać wcześniejszego użycia OCR.
- Przed publikacją repozytorium sprawdź, czy dokumenty PDF nie zawierają prywatnych lub poufnych danych.