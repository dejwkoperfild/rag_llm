from komunikacja_model import query_rag, setup_vector_db

if __name__ == "__main__":
    pdf_file = "input/sprawozdanie_fizyka.pdf"  # Podaj ścieżkę do swojego PDF-a
    print("Indeksowanie dokumentu...")
    db_collection = setup_vector_db(pdf_file)

    pytanie = "Jaki jest wzór na moc wiatru?"
    print(f"\nPytanie: {pytanie}\nOdpowiedź:")
    query_rag(db_collection, pytanie)