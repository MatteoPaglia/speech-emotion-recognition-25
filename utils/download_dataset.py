import os
import shutil
import kagglehub
from pathlib import Path

# --- CONFIGURAZIONE PERCORSI CACHE ---
def get_kaggle_cache_dir():
    """
    Ritorna il percorso standard della cache di kagglehub.
    Funziona su Windows, Linux, e macOS.
    
    Returns:
        Path: Percorso della cache di kagglehub
    """
    return Path.home() / '.cache' / 'kagglehub' / 'datasets'


def get_dataset_path(dataset_name):
    """
    Ritorna il percorso standardizzato di un dataset nella cache kagglehub.
    Garantisce un percorso consistente indipendentemente dal contesto di esecuzione.
    
    Args:
        dataset_name (str): Nome del dataset ('ravdess' o 'iemocap')
        
    Returns:
        Path: Percorso assoluto del dataset nella cache
    """
    dataset_mapping = {
        'ravdess': 'ravdess-emotional-speech-audio',
        'iemocap': 'iemocapfullrelease'
    }
    
    if dataset_name.lower() not in dataset_mapping:
        raise ValueError(f"Dataset sconosciuto: {dataset_name}")
    
    cache_dir = get_kaggle_cache_dir()
    dataset_folder = dataset_mapping[dataset_name.lower()]
    
    # La cache di kagglehub ha struttura:
    # ~/.cache/kagglehub/datasets/<username>/<dataset_slug>/<version>/
    # Cerchiamo il primo match
    if cache_dir.exists():
        for user_dir in cache_dir.iterdir():
            if user_dir.is_dir():
                for dataset_dir in user_dir.iterdir():
                    if dataset_dir.is_dir() and dataset_folder in dataset_dir.name:
                        # Ritorna la cartella che contiene i dati effettivi
                        versions = list(dataset_dir.glob('*/'))
                        if versions:
                            return versions[0]  # Prendi la prima versione
    
    return None


def setup_kaggle():
    """
    Cerca il file kaggle.json nella stessa cartella dello script (utils)
    e lo configura in /root/.kaggle/ per l'autenticazione.
    """
    print("--- 1. Configurazione Kaggle ---")
    
    # Percorso della cartella dove si trova questo script .py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Percorso del file json sorgente
    json_source = os.path.join(script_dir, 'kaggle.json')
    
    # Percorso di destinazione nel sistema
    target_dir = '/root/.kaggle'
    target_file = os.path.join(target_dir, 'kaggle.json')

    if not os.path.exists(json_source):
        print(f"ERRORE: File 'kaggle.json' non trovato in: {json_source}")
        return False

    os.makedirs(target_dir, exist_ok=True)

    try:
        shutil.copy(json_source, target_file)
        os.chmod(target_file, 0o600)  # Permessi richiesti da Kaggle
        print("Kaggle configurato con successo.")
        return True
    except Exception as e:
        print(f"Errore configurazione Kaggle: {e}")
        return False


def download_dataset_via_hub(dataset_slug, target_folder_name, filter_keyword=None):
    """
    Funzione generica che scarica un dataset tramite kagglehub.
    Il dataset rimane nella cache di sistema e non viene copiato.
    """
    print(f"\n--- Download {target_folder_name.upper()} ---")
    
    try:
        # 1. Scarica (o recupera dalla cache)
        print(f"Contatto KaggleHub per scaricare: {dataset_slug}...")
        cached_path = kagglehub.dataset_download(dataset_slug)
        print(f"✓ Dataset scaricato nella cache di sistema: {cached_path}")

        # 2. Verifica che i file esistano
        if not os.path.exists(cached_path) or not os.listdir(cached_path):
            print(f"ERRORE: La directory scaricata è vuota o non esiste")
            return False

        # 3. Verifica finale
        num_files = sum([len(files) for _, _, files in os.walk(cached_path)])
        print(f"{target_folder_name.upper()} pronto in cache: {cached_path}")
        print(f"Numero totale di file: {num_files}")
        return True
        
    except Exception as e:
        print(f"Errore durante il download di {target_folder_name}: {e}")
        return False


def download_ravdess():
    return download_dataset_via_hub("uwrfkaggler/ravdess-emotional-speech-audio", "ravdess")


def download_iemocap():
    return download_dataset_via_hub("dejolilandry/iemocapfullrelease", "iemocap")


if __name__ == "__main__":
    if setup_kaggle():
        # Download RAVDESS
        ravdess_ok = download_ravdess()
        # Download IEMOCAP
        iemocap_ok = download_iemocap()
        
        # Riepilogo finale
        print("\n" + "="*60)
        print("RIEPILOGO DOWNLOAD")
        print("="*60)
        print(f"RAVDESS: {'✅ Successo' if ravdess_ok else '❌ Fallito'}")
        print(f"IEMOCAP: {'✅ Successo' if iemocap_ok else '❌ Fallito'}")
        print("="*60)
        
        if ravdess_ok and iemocap_ok:
            print("\n🎉 Tutti i dataset sono stati scaricati con successo!")
        else:
            print("\n⚠️  Alcuni download sono falliti. Controlla i messaggi sopra.")
    else:
        print("\n❌ Impossibile procedere ai download: setup Kaggle fallito.")
        print("Assicurati che il file 'kaggle.json' sia nella cartella corretta.")