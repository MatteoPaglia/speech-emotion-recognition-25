"""
Utility functions per la gestione consistente dei percorsi dataset.
Garantisce che i dataset vengono sempre cercati nella stessa cartella cache.
"""

from pathlib import Path
import os


def get_kaggle_cache_dir():
    """
    Ritorna il percorso standard della cache di kagglehub.
    Funziona su Windows, Linux, e macOS.
    
    Returns:
        Path: Percorso della cache di kagglehub
    """
    return Path.home() / '.cache' / 'kagglehub' / 'datasets'


def find_dataset_in_cache(dataset_name):
    """
    Cerca un dataset nella cache di kagglehub in modo robusto.
    
    Args:
        dataset_name (str): Nome del dataset ('ravdess' o 'iemocap')
        
    Returns:
        Path: Percorso assoluto del dataset se trovato, None altrimenti
    """
    dataset_mapping = {
        'ravdess': 'ravdess-emotional-speech-audio',
        'iemocap': 'iemocapfullrelease'
    }
    
    dataset_key = dataset_name.lower()
    if dataset_key not in dataset_mapping:
        raise ValueError(f"Dataset sconosciuto: {dataset_name}. Usa 'ravdess' o 'iemocap'")
    
    dataset_folder_name = dataset_mapping[dataset_key]
    cache_dir = get_kaggle_cache_dir()
    
    if not cache_dir.exists():
        return None
    
    # Struttura della cache:
    # ~/.cache/kagglehub/datasets/<username>/<dataset_slug>/<version>/
    # Cerchiamo il primo match che contiene il dataset_folder_name
    try:
        for user_dir in cache_dir.iterdir():
            if user_dir.is_dir():
                for dataset_dir in user_dir.iterdir():
                    if dataset_dir.is_dir() and dataset_folder_name in dataset_dir.name:
                        # Prendi la prima versione disponibile
                        versions = sorted([d for d in dataset_dir.iterdir() if d.is_dir()], reverse=True)
                        if versions:
                            dataset_path = versions[0]
                            # Verifica che il dataset abbia file
                            if list(dataset_path.glob('*')):
                                return dataset_path
    except Exception as e:
        print(f"Errore durante ricerca di {dataset_name} nella cache: {e}")
    
    return None


def find_dataset_paths(datasets=['ravdess', 'iemocap']):
    """
    Ricerca i percorsi di uno o più dataset nella cache di kagglehub.
    
    Args:
        datasets (list): Lista di dataset da cercare ('ravdess' e/o 'iemocap')
        
    Returns:
        dict: Dizionario con i percorsi trovati {dataset_name: path}
    """
    paths = {}
    
    for dataset_name in datasets:
        path = find_dataset_in_cache(dataset_name)
        if path:
            paths[dataset_name] = path
            print(f"✅ {dataset_name.upper()} trovato: {path}")
        else:
            print(f"❌ {dataset_name.upper()} NON trovato nella cache")
    
    return paths


def validate_dataset(dataset_path, dataset_name='unknown'):
    """
    Verifica che un dataset esista e contenga file.
    
    Args:
        dataset_path (Path): Percorso del dataset
        dataset_name (str): Nome del dataset (per messaggi di debug)
        
    Returns:
        bool: True se il dataset è valido, False altrimenti
    """
    if not dataset_path or not Path(dataset_path).exists():
        return False
    
    dataset_path = Path(dataset_path)
    if not list(dataset_path.glob('*')):
        return False
    
    return True
