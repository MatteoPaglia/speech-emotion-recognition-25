#!/usr/bin/env python
"""
Fix Dataset Extraction
=======================

Script per estrarre manualmente i file .zip dei dataset se l'unzip automatico ha fallito.

Utilizzo:
    python fix_dataset_extraction.py
"""

import os
import sys
import zipfile
from pathlib import Path

# ============================================================================
# SETUP PERCORSI
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

# ============================================================================
# FUNZIONI
# ============================================================================

def extract_zip(zip_path, extract_to):
    """
    Estrae un file ZIP in una cartella di destinazione.
    """
    print(f"\n📦 Estrazione: {zip_path.name}")
    print(f"   Destinazione: {extract_to}")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            total_files = len(zip_ref.namelist())
            print(f"   File da estrarre: {total_files}")
            
            for i, file in enumerate(zip_ref.namelist(), 1):
                zip_ref.extract(file, extract_to)
                if i % 100 == 0:
                    print(f"   Estratti {i}/{total_files} file...")
            
            print(f"   ✅ Estrazione completata!")
        
        return True
    except Exception as e:
        print(f"   ❌ Errore durante l'estrazione: {e}")
        return False


def main():
    """
    Controlla e estrae i dataset.
    """
    
    print("\n" + "="*80)
    print("🔧 FIX DATASET EXTRACTION")
    print("="*80 + "\n")
    
    # ========================================================================
    # RAVDESS
    # ========================================================================
    ravdess_dir = DATA_DIR / "ravdess"
    ravdess_zip = ravdess_dir / "ravdess-emotional-speech-audio.zip"
    
    print("[1] Controllo RAVDESS...")
    if ravdess_zip.exists():
        print(f"   Trovato ZIP: {ravdess_zip.name}")
        if extract_zip(ravdess_zip, ravdess_dir):
            print(f"   Rimozione ZIP dopo estrazione...")
            ravdess_zip.unlink()
            print(f"   ✅ RAVDESS OK\n")
        else:
            print(f"   ❌ Estrazione RAVDESS fallita\n")
    else:
        num_files = sum([len(files) for _, _, files in os.walk(ravdess_dir)]) if ravdess_dir.exists() else 0
        print(f"   ✅ RAVDESS già estratto ({num_files} file)\n")
    
    # ========================================================================
    # IEMOCAP
    # ========================================================================
    iemocap_dir = DATA_DIR / "iemocap"
    iemocap_zip = iemocap_dir / "iemocapfullrelease.zip"
    
    print("[2] Controllo IEMOCAP...")
    if iemocap_zip.exists():
        print(f"   Trovato ZIP: {iemocap_zip.name}")
        print(f"   ⚠️  IEMOCAP è un file molto grande (~10GB+)")
        print(f"   L'estrazione potrebbe richiedere 10-30 minuti...")
        
        response = input("   Vuoi procedere con l'estrazione? (s/n): ").lower().strip()
        if response == 's':
            if extract_zip(iemocap_zip, iemocap_dir):
                print(f"   Rimozione ZIP dopo estrazione...")
                iemocap_zip.unlink()
                print(f"   ✅ IEMOCAP OK\n")
            else:
                print(f"   ❌ Estrazione IEMOCAP fallita\n")
        else:
            print(f"   ⏭️  Saltato\n")
    else:
        num_files = sum([len(files) for _, _, files in os.walk(iemocap_dir)]) if iemocap_dir.exists() else 0
        print(f"   ✅ IEMOCAP già estratto ({num_files} file)\n")
    
    # ========================================================================
    # RIEPILOGO
    # ========================================================================
    print("="*80)
    print("📊 RIEPILOGO STATO DATASET")
    print("="*80)
    
    for dataset_name, dataset_path in [("RAVDESS", ravdess_dir), ("IEMOCAP", iemocap_dir)]:
        if dataset_path.exists():
            num_files = sum([len(files) for _, _, files in os.walk(dataset_path)])
            num_zips = len(list(dataset_path.glob("*.zip")))
            
            if num_zips > 0:
                status = f"⚠️  ZIP rimasto ({num_zips} file)"
            elif num_files > 0:
                status = f"✅ Estratto ({num_files} file)"
            else:
                status = "❓ Vuoto"
            
            print(f"{dataset_name:10} → {status}")
            print(f"             Percorso: {dataset_path}")
        else:
            print(f"{dataset_name:10} → ❌ Non trovato")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
