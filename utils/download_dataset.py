import os
import shutil

def setup_kaggle():
    """
    Cerca il file kaggle.json nella stessa cartella di questo script (utils),
    lo sposta in /root/.kaggle/ e imposta i permessi corretti.
    """
    print("--- Configurazione Kaggle ---")
    
    # 1. Ottieni il percorso assoluto della cartella dove si trova questo script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Cerca kaggle.json lì dentro
    json_source_path = os.path.join(script_dir, 'kaggle.json')
    print(f"Cerco 'kaggle.json' in: {json_source_path}")

    if not os.path.exists(json_source_path):
        print(f"ERRORE: File non trovato in {json_source_path}")
        print("Assicurati di aver trascinato kaggle.json nella cartella 'utils'!")
        return False

    # 3. Prepara la destinazione di sistema
    target_dir = '/root/.kaggle'
    target_file = os.path.join(target_dir, 'kaggle.json')
    
    os.makedirs(target_dir, exist_ok=True)

    # 4. Copia il file e imposta i permessi
    try:
        shutil.copy(json_source_path, target_file)
        # Imposta permessi di lettura/scrittura solo per l'owner (richiesto da Kaggle)
        os.chmod(target_file, 0o600)
        print("Configurazione completata! File copiato e permessi impostati.")
        return True
    except Exception as e:
        print(f"Errore durante la configurazione: {e}")
        return False

def download_ravdess():
    """
    Scarica il dataset RAVDESS da Kaggle e lo estrae.
    """
    print("\n--- Download RAVDESS ---")
    # Scarica
    exit_code = os.system('kaggle datasets download -d uwrfkaggler/ravdess-emotional-speech-audio')
    
    if exit_code == 0:
        print("Download completato. Estrazione in corso...")
        # Estrai nella cartella ./ravdess
        os.system('unzip -q -o ravdess-emotional-speech-audio.zip -d ./ravdess')
        
        # Rimuovi lo zip per risparmiare spazio (opzionale)
        # os.remove('ravdess-emotional-speech-audio.zip') 
        print("Dataset RAVDESS pronto in ./ravdess")
    else:
        print("Errore durante il download di RAVDESS.")

def download_iemocap():
    """
    Scarica il dataset IEMOCAP da Kaggle e lo estrae.
    """
    print("\n--- Download IEMOCAP ---")
    # Scarica
    #exit_code = os.system('kaggle datasets download -d mrmorj/iemocap')
    exit_code = os.system('kaggle datasets download -d l33tc0d3r/iemocap-full-release')
    
    if exit_code == 0:
        print("Download completato. Estrazione in corso...")
        # Estrai nella cartella ./iemocap
        os.system('unzip -q -o iemocap.zip -d ./iemocap')
        
        # Rimuovi lo zip per risparmiare spazio (opzionale)
        # os.remove('iemocap.zip')
        print("Dataset IEMOCAP pronto in ./iemocap")
    else:
        print("Errore durante il download di IEMOCAP.")

if __name__ == "__main__":
    # Esegui il setup, se va a buon fine scarica i dataset
    if setup_kaggle():
        download_ravdess()
        download_iemocap()
    else:
        print("Impossibile procedere con i download a causa di errori nel setup.")






