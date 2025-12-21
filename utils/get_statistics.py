from collections import Counter

def print_dataset_stats(dataset, name="DATASET"):
    """Stampa report statistico dettagliato per un dataset."""
    print(f"\n{'='*40}")
    print(f"📊 ANALISI {name.upper()}")
    print(f"{'='*40}")
    
    total = len(dataset)
    if total == 0:
        print("⚠️ Dataset vuoto!")
        return

    # 1. Estrazione dati sicura
    # Usiamo int() qui per garantire che 'actor' sia un numero,
    # prevenendo l'errore "string formatting" durante il modulo %
    try:
        actors = set(int(s['metadata']['actor']) for s in dataset.samples)
        emotions = [s['metadata']['emotion_label'] for s in dataset.samples]
    except KeyError:
        # Fallback se la struttura di 'samples' è diversa (es. IEMOCAP o versioni vecchie)
        # Prova ad accedere direttamente alle chiavi se non sono in 'metadata'
        actors = set(int(s['actor_id']) if 'actor_id' in s else 0 for s in dataset.samples)
        emotions = [s['emotion'] if 'emotion' in s else 'unknown' for s in dataset.samples]

    # 2. Statistiche Attori (Genere)
    # Ora 'a' è sicuramente int, quindi % funziona matematicamente
    males = [a for a in actors if a % 2 == 1]
    females = [a for a in actors if a % 2 == 0]
    
    print(f"🔹 Samples Totali: {total}")
    print(f"🔹 Attori ({len(actors)}): {sorted(list(actors))}")
    print(f"   - Maschi:  {len(males)}")
    print(f"   - Femmine: {len(females)}")
    
    # 3. Distribuzione Emozioni
    print(f"\n🎭 Distribuzione Emozioni:")
    counts = Counter(emotions)
    for emo, count in sorted(counts.items()):
        perc = (count / total) * 100
        # Calcolo barra visiva
        bar = "█" * int(perc / 5) 
        # F-string sicura
        print(f"   - {emo.capitalize():10s}: {count:4d} ({perc:5.1f}%) {bar}")
        
    print("-" * 40)