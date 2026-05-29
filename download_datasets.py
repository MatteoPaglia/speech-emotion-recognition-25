#!/usr/bin/env python
"""
Download Dataset Script
========================

Script standalone per scaricare RAVDESS e IEMOCAP nelle cartelle predisposte.

Prerequisiti:
- kaggle-cli: pip install kaggle
- kaggle.json nella cartella utils/

Utilizzo:
    python download_datasets.py

"""

import os
import sys
from pathlib import Path

# ============================================================================
# SETUP PERCORSI
# ============================================================================

# Determina il percorso del progetto
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
UTILS_DIR = PROJECT_ROOT / "utils"

# Aggiungi il progetto al path di Python
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Esegue il download di entrambi i dataset.
    """
    
    print("\n" + "="*80)
    print("🎵 SPEECH EMOTION RECOGNITION - DATASET DOWNLOAD")
    print("="*80 + "\n")
    
    # Verifica che kaggle.json esista
    kaggle_json = UTILS_DIR / "kaggle.json"
    if not kaggle_json.exists():
        print(f"❌ ERRORE: File 'kaggle.json' non trovato in {kaggle_json}")
        print("\nPer usare questo script:")
        print("1. Scarica il file kaggle.json da: https://www.kaggle.com/account")
        print("2. Posizionalo in: utils/kaggle.json")
        print("3. Esegui di nuovo questo script")
        sys.exit(1)
    
    # Importa le funzioni di download
    try:
        from utils.download_dataset_local import dowload_ravdess_local, dowload_iemocap_local
    except ImportError as e:
        print(f"❌ Errore nell'importazione: {e}")
        sys.exit(1)
    
    # ========================================================================
    # SCARICA RAVDESS
    # ========================================================================
    print("[1/2] Scaricamento RAVDESS...")
    print("-" * 80)
    ravdess_path = dowload_ravdess_local()
    
    if ravdess_path:
        print(f"✅ RAVDESS scaricato con successo")
        print(f"   Percorso: {ravdess_path}\n")
    else:
        print(f"❌ Errore nel download di RAVDESS\n")
    
    # ========================================================================
    # SCARICA IEMOCAP
    # ========================================================================
    print("[2/2] Scaricamento IEMOCAP...")
    print("-" * 80)
    iemocap_path = dowload_iemocap_local()
    
    if iemocap_path:
        print(f"✅ IEMOCAP scaricato con successo")
        print(f"   Percorso: {iemocap_path}\n")
    else:
        print(f"❌ Errore nel download di IEMOCAP\n")
    
    # ========================================================================
    # RIEPILOGO
    # ========================================================================
    print("="*80)
    print("📊 RIEPILOGO DOWNLOAD")
    print("="*80)
    
    ravdess_status = "✅ Completato" if ravdess_path else "❌ Fallito"
    iemocap_status = "✅ Completato" if iemocap_path else "❌ Fallito"
    
    print(f"RAVDESS:  {ravdess_status}")
    if ravdess_path:
        num_files = sum([len(files) for _, _, files in os.walk(ravdess_path)])
        print(f"          → {ravdess_path}")
        print(f"          → File totali: {num_files}")
    
    print(f"\nIEMOCAP:  {iemocap_status}")
    if iemocap_path:
        num_files = sum([len(files) for _, _, files in os.walk(iemocap_path)])
        print(f"          → {iemocap_path}")
        print(f"          → File totali: {num_files}")
    
    print("="*80)
    
    # ========================================================================
    # PROSSIMI PASSI
    # ========================================================================
    if ravdess_path and iemocap_path:
        print("\n✅ Entrambi i dataset sono pronti!")
        print("\n📝 Prossimi passi:")
        print("   1. Esegui il notebook: ravdess_train/SpeechEmotionRecnognition_Local.ipynb")
        print("   2. Oppure esegui il training con: python ravdess_train/train.py")
        print("\n" + "="*80 + "\n")
        return 0
    else:
        print("\n❌ Alcuni dataset non sono stati scaricati correttamente.")
        print("   Verifica i log sopra e riprova.\n")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
