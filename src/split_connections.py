import json
import os
import random
from sklearn.model_selection import train_test_split

# CONFIGURAZIONE
INPUT_FILE = "data/processed/hf_kaggle_connections_quiz.jsonl" # Assicurati che il file unito sia qui
OUTPUT_DIR = "data/processed"
SEED = 42

def split_dataset():
    print(f"Caricamento puzzle da {INPUT_FILE}...")
    
    if not os.path.exists(INPUT_FILE):
        print(f"ERRORE: Non trovo {INPUT_FILE}. Assicurati di aver unito i dataset.")
        return

    # Impostiamo il seed per riproducibilità (sia per lo split che per lo shuffle delle parole)
    random.seed(SEED)

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        # Legge riga per riga ignorando eventuali righe vuote
        all_puzzles = [json.loads(line) for line in f if line.strip()]

    # Rimozione duplicati basata su puzzle_id
    unique_puzzles_dict = {p['puzzle_id']: p for p in all_puzzles}
    unique_puzzles = list(unique_puzzles_dict.values())
    
    print(f"Totale puzzle unici trovati: {len(unique_puzzles)}")
    
    # Estraiamo gli ID per lo split
    ids = [p['puzzle_id'] for p in unique_puzzles]

    # Split 1: Train (80%) vs Temp (20%)
    train_ids, temp_ids = train_test_split(ids, test_size=0.20, random_state=SEED)
    
    # Split 2: Temp -> Validation (10%) + Test (10%)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.50, random_state=SEED)

    print(f"Split definitivi -> Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

    # Creiamo i set di ID per lookup veloce
    train_set_ids = set(train_ids)
    val_set_ids = set(val_ids)
    test_set_ids = set(test_ids)

    # Ripartiamo i dati completi
    train_data = [p for p in unique_puzzles if p['puzzle_id'] in train_set_ids]
    val_data = [p for p in unique_puzzles if p['puzzle_id'] in val_set_ids]
    test_data = [p for p in unique_puzzles if p['puzzle_id'] in test_set_ids]

    # --- MODIFICA RICHIESTA: SHUFFLE DELLE PAROLE ---
    def shuffle_words_in_puzzles(dataset, label):
        print(f"Eseguo shuffle delle parole per il set: {label}...")
        for puzzle in dataset:
            # Shuffle in-place della lista words
            # Importante: questo non rompe la soluzione perché le answers si basano sulle stringhe, non sugli indici
            random.shuffle(puzzle['words'])
        return dataset

    train_data = shuffle_words_in_puzzles(train_data, "Train")
    val_data = shuffle_words_in_puzzles(val_data, "Validation")
    test_data = shuffle_words_in_puzzles(test_data, "Test")
    # -----------------------------------------------

    # Funzione helper salvataggio
    def save_jsonl(data, filename):
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, 'w', encoding='utf-8') as f:
            for entry in data:
                f.write(json.dumps(entry) + '\n')
        print(f"Salvato {filename} ({len(data)} righe)")

    save_jsonl(train_data, "connections_train.jsonl")
    save_jsonl(val_data, "connections_val.jsonl")
    save_jsonl(test_data, "connections_test.jsonl")

if __name__ == "__main__":
    split_dataset()