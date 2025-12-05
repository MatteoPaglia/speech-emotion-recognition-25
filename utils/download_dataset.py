import os
# Importiamo google.colab solo se siamo in quell'ambiente per evitare errori locali
try:
    from google.colab import files
except ImportError:
    print("Nota: google.colab non trovato. Questo script deve girare su Colab.")

def setup_kaggle():
    """
    Carica il file kaggle.json su Colab e configura l'ambiente.
    """
    print("Carica il file kaggle.json (scaricato dal tuo account Kaggle)...")
    
    # Questo aprirà il widget di upload su Colab
    uploaded = files.upload() 

    # Verifica se il file è stato caricato
    if 'kaggle.json' in uploaded:
        # Crea la cartella se non esiste
        os.makedirs('/root/.kaggle', exist_ok=True)
        
        # Sposta il file (usiamo os.rename o os.system per compatibilità)
        # mv in python è os.rename, ma tra file system diversi meglio shutil o os.system
        os.system('mv kaggle.json /root/.kaggle/')
        
        # Imposta i permessi (chmod 600)
        os.system('chmod 600 /root/.kaggle/kaggle.json')
        print("Configurazione completata!")
    else:
        print("Errore: file kaggle.json non caricato.")

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