import ollama
import chromadb
from extract_pdf import extract_text_from_pdf




def setup_vector_db(pdf_path: str):
    """Indeksuje plik PDF w lokalnej bazie ChromaDB."""
    client = chromadb.Client()
    collection = client.get_or_create_collection(name="dokumenty_rag")

    chunks = extract_text_from_pdf(pdf_path)

    # Generowanie embeddingów przez Ollama i zapis do bazy
    for i, chunk in enumerate(chunks):
        response = ollama.embeddings(model="nomic-embed-text", prompt=chunk)
        embedding = response["embedding"]
        collection.add(
            ids=[f"chunk_{i}"],
            embeddings=[embedding],
            documents=[chunk],
        )

    return collection


def query_rag(collection, question: str, n_results: int = 3):
    """Szuka najbardziej pasujących fragmentów i pyta model."""
    # 1. Wektoryzacja pytania
    query_embed = ollama.embeddings(
        model="nomic-embed-text", prompt=question
    )["embedding"]

    # 2. Wyszukanie w ChromaDB
    results = collection.query(query_embeddings=[query_embed], n_results=n_results)
    retrieved_chunks = results["documents"][0]
    context = "\n---\n".join(retrieved_chunks)

    # 3. Złożenie promptu z kontekstem
    prompt = f"""Odpowiedz na pytanie na podstawie poniższego kontekstu. 
Jeśli informacja nie znajduje się w kontekście, napisz wprost, że tego nie wiesz.

Kontekst:
{context}

Pytanie: {question}
Odpowiedź:"""

    # 4. Generowanie odpowiedzi
    stream = ollama.chat(
        model="SpeakLeash/bielik-minitron-7B-v3.0-instruct:Q4_K_M",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    for chunk in stream:
        print(chunk["message"]["content"], end="", flush=True)
    print()
