#!/usr/bin/env python
"""
Download e Estrai IEMOCAP
==========================

Script per scaricare IEMOCAP da Kaggle e estrarlo automaticamente con progress bar.

Prerequisiti:
- kaggle-cli: pip install kaggle
- tqdm: pip install tqdm (opzionale, per progress bar)
- kaggle.json nella cartella utils/

Utilizzo:
    python download_iemocap_with_progress.py
"""

import os
import sys
import zipfile
import subprocess
from pathlib import Path

# Prova a importare tqdm per progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("⚠️  tqdm non installato. Installa con: pip install tqdm")
    print("    Per ora continuerò senza progress bar.\n")

# ============================================================================
# SETUP PERCORSI
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
UTILS_DIR = PROJECT_ROOT / "utils"
DATA_DIR = PROJECT_ROOT / "data"
IEMOCAP_DIR = DATA_DIR / "iemocap"

# ============================================================================
# FUNZIONI
# ============================================================================

def setup_kaggle():
    """
    Configura Kaggle usando il file kaggle.json nella cartella utils.
    """
    print("🔐 Configurazione Kaggle...")
    
    json_source = UTILS_DIR / 'kaggle.json'
    
    if not json_source.exists():
        print(f"❌ ERRORE: File 'kaggle.json' non trovato in: {json_source}")
        print("\nPer risolvere:")
        print("1. Vai a: https://www.kaggle.com/account")
        print("2. Clicca 'Create New API Token'")
        print("3. Salva il file kaggle.json in: utils/kaggle.json")
        return False

    import platform
    target_dir = Path.home() / '.kaggle'
    target_file = target_dir / 'kaggle.json'

    target_dir.mkdir(exist_ok=True)

    try:
        import shutil
        shutil.copy(json_source, target_file)
        if platform.system() != 'Windows':
            os.chmod(target_file, 0o600)
        print("✅ Kaggle configurato correttamente\n")
        return True
    except Exception as e:
        print(f"❌ Errore: {e}")
        return False


def download_iemocap():
    """
    Scarica IEMOCAP da Kaggle usando kaggle-cli.
    """
    print("📥 Download IEMOCAP da Kaggle...")
    print("   (questo potrebbe richiedere 10-30 minuti)\n")
    
    IEMOCAP_DIR.mkdir(parents=True, exist_ok=True)
    
    # Comandi: scarica senza --unzip per gestire l'estrazione manualmente
    cmd = f'kaggle datasets download -d dejolilandry/iemocapfullrelease -p "{IEMOCAP_DIR}"'
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
        
        if result.returncode != 0:
            print(f"\n❌ Errore Kaggle")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Errore durante il download: {e}")
        return False


def extract_iemocap():
    """
    Estrae IEMOCAP con progress bar.
    """
    zip_path = IEMOCAP_DIR / "iemocapfullrelease.zip"
    
    if not zip_path.exists():
        print(f"❌ File ZIP non trovato: {zip_path}")
        return False
    
    print(f"\n📦 Estrazione IEMOCAP...")
    print(f"   ZIP: {zip_path.name}")
    print(f"   Destinazione: {IEMOCAP_DIR}")
    print(f"   (l'estrazione potrebbe richiedere 5-15 minuti)\n")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            total_files = len(file_list)
            
            # Estrai con o senza progress bar
            if HAS_TQDM:
                print(f"   Totale file: {total_files}\n")
                for file in tqdm(file_list, desc="Estrazione", unit="file", ncols=80):
                    zip_ref.extract(file, IEMOCAP_DIR)
            else:
                print(f"   Totale file: {total_files}")
                for i, file in enumerate(file_list, 1):
                    zip_ref.extract(file, IEMOCAP_DIR)
                    if i % 500 == 0:
                        print(f"   {i}/{total_files} file estratti...")
        
        print(f"\n✅ Estrazione completata!")
        return True
    
    except Exception as e:
        print(f"❌ Errore durante l'estrazione: {e}")
        return False


def cleanup_zip():
    """
    Elimina il file ZIP dopo l'estrazione.
    """
    zip_path = IEMOCAP_DIR / "iemocapfullrelease.zip"
    
    try:
        if zip_path.exists():
            print(f"\n🧹 Rimozione ZIP dopo estrazione...")
            zip_path.unlink()
            print(f"   ✅ ZIP eliminato")
        return True
    except Exception as e:
        print(f"⚠️  Errore durante l'eliminazione: {e}")
        return False


def verify_iemocap():
    """
    Verifica che IEMOCAP sia stato estratto correttamente.
    """
    print(f"\n🔍 Verifica dataset...")
    
    # Cerca la cartella IEMOCAP_full_release
    iemocap_full_release = IEMOCAP_DIR / "IEMOCAP_full_release"
    
    if iemocap_full_release.exists():
        # Conta file e cartelle
        num_files = sum([len(files) for _, _, files in os.walk(iemocap_full_release)])
        num_sessions = len(list(iemocap_full_release.glob("Session*")))
        
        print(f"   ✅ IEMOCAP estratto correttamente!")
        print(f"   Percorso: {iemocap_full_release}")
        print(f"   Session trovate: {num_sessions}")
        print(f"   File totali: {num_files}")
        
        return str(iemocap_full_release)
    else:
        print(f"   ⚠️  Cartella IEMOCAP_full_release non trovata")
        print(f"   Controllo in: {IEMOCAP_DIR}")
        
        # Elenca cosa c'è
        if IEMOCAP_DIR.exists():
            contents = list(IEMOCAP_DIR.iterdir())
            print(f"   Contenuto della cartella:")
            for item in contents[:10]:
                print(f"      - {item.name}")
        
        return None


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Funzione principale.
    """
    
    print("\n" + "="*80)
    print("🎵 DOWNLOAD IEMOCAP CON PROGRESS BAR")
    print("="*80 + "\n")
    
    # ========================================================================
    # Step 1: Configura Kaggle
    # ========================================================================
    if not setup_kaggle():
        print("\n❌ Impossibile procedere senza kaggle.json")
        return 1
    
    # ========================================================================
    # Step 2: Scarica IEMOCAP
    # ========================================================================
    if not download_iemocap():
        print("\n❌ Download fallito")
        return 1
    
    # ========================================================================
    # Step 3: Estrai IEMOCAP
    # ========================================================================
    if not extract_iemocap():
        print("\n❌ Estrazione fallita")
        return 1
    
    # ========================================================================
    # Step 4: Pulizia
    # ========================================================================
    cleanup_zip()
    
    # ========================================================================
    # Step 5: Verifica
    # ========================================================================
    iemocap_path = verify_iemocap()
    
    # ========================================================================
    # Riepilogo finale
    # ========================================================================
    print("\n" + "="*80)
    print("✅ DOWNLOAD E ESTRAZIONE COMPLETATI!")
    print("="*80)
    
    if iemocap_path:
        print(f"\n📁 Percorso IEMOCAP: {iemocap_path}")
        print(f"\n📝 Prossimi passi:")
        print(f"   1. Esegui il notebook:")
        print(f"      ravdess_train/SpeechEmotionRecnognition_Local.ipynb")
        print(f"   2. Oppure esegui il training direttamente:")
        print(f"      python ravdess_train/train.py --dataset_path \"{iemocap_path}\"")
        print("\n" + "="*80 + "\n")
        return 0
    else:
        print("\n❌ Verifica fallita")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
