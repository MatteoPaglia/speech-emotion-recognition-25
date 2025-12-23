from collections import Counter

def print_dataset_stats(samples, name="DATASET"):
    """
    Stampa report statistico dettagliato per un dataset.
    Supporta sia la struttura RAVDESS che IEMOCAP.
    
    Args:
        samples: Lista di sample dict (oppure oggetto Dataset)
        name: Nome del dataset per la stampa
    """
    print(f"\n{'='*40}")
    print(f"📊 ANALISI {name.upper()}")
    print(f"{'='*40}")
    
    # Se passa un oggetto Dataset, estrai i samples
    if hasattr(samples, 'samples'):
        samples = samples.samples
    
    total = len(samples)
    if total == 0:
        print("⚠️ Dataset vuoto!")
        return

    # 1. Rileva il tipo di dataset dalla struttura del primo sample
    if total == 0:
        return
    
    first_sample = samples[0]
    is_ravdess = 'metadata' in first_sample  # RAVDESS ha 'metadata'
    
    # 2. Estrazione dati basata sul tipo di dataset
    actors = set()
    emotions = []
    
    for s in samples:
        try:
            if is_ravdess:
                # Struttura RAVDESS
                actor = int(s['metadata']['actor'])
                emotion = s['metadata']['emotion_label']
            else:
                # Struttura IEMOCAP
                actor_str = s['actor']
                # Converti M/F a numero (M=1, F=2) per compatibilità con modulo
                actor = 1 if actor_str == 'M' else 2
                # Mappa il codice emozione al nome
                emotion_map = {'neu': 'neutral', 'hap': 'happy', 'sad': 'sad', 'ang': 'angry'}
                emotion = emotion_map.get(s['label'], s['label'])
            
            actors.add(actor)
            emotions.append(emotion)
        except (KeyError, ValueError, TypeError) as e:
            continue
    
    # 3. Statistiche Attori (Genere)
    males = [a for a in actors if a % 2 == 1]
    females = [a for a in actors if a % 2 == 0]
    
    print(f"🔹 Samples Totali: {total}")
    print(f"🔹 Attori ({len(actors)}): {sorted(list(actors))}")
    print(f"   - Maschi:  {len(males)}")
    print(f"   - Femmine: {len(females)}")
    
    # 4. Distribuzione Emozioni
    print(f"\n🎭 Distribuzione Emozioni:")
    counts = Counter(emotions)
    for emo, count in sorted(counts.items()):
        perc = (count / total) * 100 if total > 0 else 0
        # Calcolo barra visiva
        bar = "█" * int(perc / 5) 
        # F-string sicura
        print(f"   - {emo.capitalize():10s}: {count:4d} ({perc:5.1f}%) {bar}")
        
    print("-" * 40)