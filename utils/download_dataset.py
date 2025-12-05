import os
# Importiamo google.colab solo se siamo in quell'ambiente per evitare errori locali
try:
    from google.colab import files
except ImportError:
    print("Nota: google.colab non trovato. Questo script deve girare su Colab.")

import os
import shutil

def setup_kaggle():
    """
    Configura Kaggle assumendo che il file kaggle.json sia già stato caricato
    manualmente nella cartella del progetto.
    """
    json_file = 'kaggle.json'
    target_dir = '/root/.kaggle'
    
    # 1. Controlla se il file esiste nella directory corrente (dove hai fatto il drag & drop)
    if not os.path.exists(json_file):
        print(f"ERRORE: Il file '{json_file}' non è stato trovato nella cartella corrente.")
        print("Per favore, trascina 'kaggle.json' dentro la cartella del progetto in VS Code prima di eseguire lo script.")
        return

    print(f"File '{json_file}' trovato! Procedo alla configurazione...")

    # 2. Crea la directory di destinazione
    os.makedirs(target_dir, exist_ok=True)

    # 3. Sposta il file (usiamo shutil per sicurezza tra file system)
    try:
        shutil.copy(json_file, os.path.join(target_dir, json_file))
        print(f"File copiato in {target_dir}")
    except Exception as e:
        print(f"Errore durante la copia: {e}")

    # 4. Imposta i permessi
    try:
        os.chmod(os.path.join(target_dir, json_file), 0o600)
        print("Permessi impostati correttamente (600).")
        print("Configurazione completata!")
    except Exception as e:
        print(f"Errore impostazione permessi: {e}")

# ... resto delle funzioni download_ravdess e download_iemocap uguali a prima ...

def download_ravdess():
    """
    Scarica ed estrae RAVDESS.
    """
    print("Scaricando il dataset RAVDESS...")
    # Sostituisci ! con os.system
    exit_code = os.system('kaggle datasets download -d uwrfkaggler/ravdess-emotional-speech-audio')
    
    if exit_code == 0:
        print("Download completato. Estrazione in corso...")
        os.system('unzip -q ravdess-emotional-speech-audio.zip -d ./ravdess')
        print("Estrazione completata!")
    else:
        print("Errore durante il download di RAVDESS.")

def download_iemocap():
    """
    Scarica ed estrae IEMOCAP.
    """
    print("Scaricando il dataset IEMOCAP...")
    exit_code = os.system('kaggle datasets download -d mrmorj/iemocap')
    
    if exit_code == 0:
        print("Download completato. Estrazione in corso...")
        os.system('unzip -q iemocap.zip -d ./iemocap')
        print("Estrazione completata!")
    else:
        print("Errore durante il download di IEMOCAP.")

if __name__ == "__main__":
    setup_kaggle()
    download_ravdess()
    download_iemocap()