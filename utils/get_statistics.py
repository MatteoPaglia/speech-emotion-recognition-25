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

    # Estrazione dati
    actors = set(s['metadata']['actor'] for s in dataset.samples)
    emotions = [s['metadata']['emotion_label'] for s in dataset.samples]
    
    # Statistiche Attori
    males = [a for a in actors if a % 2 == 1]
    females = [a for a in actors if a % 2 == 0]
    
    print(f"🔹 Samples Totali: {total}")
    print(f"🔹 Attori ({len(actors)}): {sorted(list(actors))}")
    print(f"   - Maschi:  {len(males)}")
    print(f"   - Femmine: {len(females)}")
    
    # Distribuzione Emozioni
    print(f"\n🎭 Distribuzione Emozioni:")
    counts = Counter(emotions)
    for emo, count in sorted(counts.items()):
        perc = (count / total) * 100
        bar = "█" * int(perc / 5)
        print(f"   - {emo.capitalize():10s}: {count:4d} ({perc:5.1f}%) {bar}")
    print("-" * 40)

