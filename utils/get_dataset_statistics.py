from collections import Counter

def print_dataset_stats(samples, name="DATASET"):
    """
    Stampa report statistico dettagliato per RAVDESS.
    
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

    # Estrazione dati RAVDESS
    actors = set()
    emotions = []
    
    for s in samples:
        try:
            # Struttura RAVDESS: metadata -> actor, emotion_label
            actor = int(s['metadata']['actor'])
            emotion = s['metadata']['emotion_label']
            
            actors.add(actor)
            emotions.append(emotion)
        except (KeyError, ValueError, TypeError) as e:
            continue
    
    # Statistiche Attori (Genere)
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
        perc = (count / total) * 100 if total > 0 else 0
        # Calcolo barra visiva
        bar = "█" * int(perc / 5) 
        # F-string sicura
        print(f"   - {emo.capitalize():10s}: {count:4d} ({perc:5.1f}%) {bar}")
        
    print("-" * 40)



def print_iemocap_stats(samples, name="IEMOCAP"):
    """
    Stampa report statistico dettagliato per il dataset IEMOCAP.
    Mostra SPEAKER INDEPENDENCE per verificare che non ci sia leakage tra split.
    Gli attori sono identificati come (session_id, gender) poiché ogni sessione ha max 2 speaker unici.
    
    Args:
        samples: Lista di sample dict (oppure oggetto Dataset con attributo .samples)
        name: Nome del dataset per la stampa
    """
    print(f"\n{'='*60}")
    print(f"📊 ANALISI {name.upper()}")
    print(f"{'='*60}")
    
    # Se passa un oggetto Dataset, estrai i samples
    if hasattr(samples, 'samples'):
        samples = samples.samples
    
    total = len(samples)
    if total == 0:
        print("⚠️ Dataset vuoto!")
        return

    # Estrazione dati IEMOCAP
    emotions = []
    sessions = set()
    improvs = set()
    
    # Struttura per tracciare gli speaker: (session_id, actor_gender)
    speakers = set()  # Insieme di tuple (session, actor)
    session_speaker_map = {}  # {session: {M, F}}
    
    # Mappa per emozioni IEMOCAP
    emotion_map = {
        'neu': 'neutral',
        'hap': 'happy',
        'exc': 'happy',      # ✅ EXCITEMENT MAPPATO A HAPPY
        'sad': 'sad',
        'ang': 'angry'
    }
    
    for s in samples:
        try:
            # Struttura IEMOCAP: session_id, actor, label, impro_id
            session = s['session_id']
            actor = s['actor']  # 'M' o 'F'
            emotion_code = s['label']
            emotion = emotion_map.get(emotion_code, emotion_code)
            impro_id = s['impro_id']
            
            sessions.add(session)
            improvs.add(impro_id)
            emotions.append(emotion)
            
            # Traccia speaker come (session, gender)
            speaker_id = (session, actor)
            speakers.add(speaker_id)
            
            # Mappa speaker per sessione
            if session not in session_speaker_map:
                session_speaker_map[session] = set()
            session_speaker_map[session].add(actor)
                
        except (KeyError, ValueError, TypeError) as e:
            print(f"⚠️ Errore nel parsing sample IEMOCAP: {e}")
            continue
    
    # === STATISTICHE GENERALI ===
    print(f"\n🔹 SAMPLES TOTALI: {total}")
    print(f"🔹 SESSIONI: {sorted(list(sessions))}")
    print(f"🔹 SPEAKER UNICI (session, gender): {len(speakers)}")
    print(f"   Elenco: {sorted(list(speakers))}")
    print(f"🔹 IMPROVVISAZIONI UNICHE: {len(improvs)}")
    
    # === SPEAKER INDEPENDENCE CHECK ===
    print(f"\n👥 SPEAKER INDEPENDENCE (per verificare leakage):")
    for session in sorted(session_speaker_map.keys()):
        genders = sorted(list(session_speaker_map[session]))
        speakers_in_session = [f"(Ses{session}, {g})" for g in genders]
        print(f"   - Sessione {session}: {', '.join(speakers_in_session)}")
    
    # === DISTRIBUZIONE EMOZIONI ===
    print(f"\n🎭 DISTRIBUZIONE EMOZIONI:")
    counts = Counter(emotions)
    for emo, count in sorted(counts.items()):
        perc = (count / total) * 100 if total > 0 else 0
        bar = "█" * int(perc / 5) 
        print(f"   - {emo.capitalize():10s}: {count:4d} ({perc:5.1f}%) {bar}")
    
    # === DISTRIBUZIONE PER SESSIONE ===
    print(f"\n📋 DISTRIBUZIONE CAMPIONI PER SESSIONE:")
    session_counts = {}
    session_emotion_counts = {}  # {session: {emotion: count}}
    
    for s in samples:
        session = s['session_id']
        emotion_code = s['label']
        emotion = emotion_map.get(emotion_code, emotion_code)
        
        session_counts[session] = session_counts.get(session, 0) + 1
        
        if session not in session_emotion_counts:
            session_emotion_counts[session] = {}
        session_emotion_counts[session][emotion] = session_emotion_counts[session].get(emotion, 0) + 1
    
    for session in sorted(session_counts.keys()):
        count = session_counts[session]
        perc = (count / total) * 100 if total > 0 else 0
        bar = "█" * int(perc / 5)
        print(f"   - Sessione {session}: {count:4d} ({perc:5.1f}%) {bar}")
        
        # Sub-distribuzione emozioni per sessione
        emo_dist = session_emotion_counts[session]
        for emo in sorted(emo_dist.keys()):
            emo_count = emo_dist[emo]
            emo_perc = (emo_count / count) * 100
            print(f"      └─ {emo.capitalize():10s}: {emo_count:3d} ({emo_perc:5.1f}%)")
        
    print("-" * 60)


