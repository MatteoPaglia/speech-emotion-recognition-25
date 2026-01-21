"""
Download dataset usando Kaggle CLI - SOLO PER USO LOCALE CON VENV
Non importa kagglehub, evita conflitti di versioni con kagglesdk.
"""
import os
import shutil
import subprocess


def setup_kaggle():
    """
    Cerca il file kaggle.json nella stessa cartella dello script (utils)
    e lo configura per l'autenticazione Kaggle.
    """
    print("--- 1. Configurazione Kaggle ---")
    
    # Percorso della cartella dove si trova questo script .py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Percorso del file json sorgente
    json_source = os.path.join(script_dir, 'kaggle.json')
    
    # Percorso di destinazione nel sistema
    # Windows: %USERPROFILE%\.kaggle
    # Linux/Mac: ~/.kaggle
    import platform
    if platform.system() == 'Windows':
        target_dir = os.path.join(os.path.expanduser('~'), '.kaggle')
    else:
        target_dir = os.path.join(os.path.expanduser('~'), '.kaggle')
    
    target_file = os.path.join(target_dir, 'kaggle.json')

    if not os.path.exists(json_source):
        print(f"ERRORE: File 'kaggle.json' non trovato in: {json_source}")
        return False

    os.makedirs(target_dir, exist_ok=True)

    try:
        shutil.copy(json_source, target_file)
        if platform.system() != 'Windows':
            os.chmod(target_file, 0o600)  # Permessi richiesti da Kaggle (Linux/Mac)
        print("✓ Kaggle configurato con successo.")
        return True
    except Exception as e:
        print(f"Errore configurazione Kaggle: {e}")
        return False


def dowload_ravdess_local():
    """
    Scarica RAVDESS localmente nella directory 'data/ravdess/' del progetto.
    Se esiste già, salta il download.
    Usa Kaggle CLI per velocità: scarica direttamente senza cache intermedia.
    
    Returns:
        str: Percorso completo del dataset scaricato se successo
        None: Se fallisce il download o setup Kaggle
    """
    print("\n--- Download RAVDESS (locale) ---")
    
    try:
        # 1. Definisci il percorso di destinazione
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        local_ravdess_path = os.path.join(project_root, 'data', 'ravdess')
        
        # 2. Verifica se esiste già
        if os.path.exists(local_ravdess_path) and os.listdir(local_ravdess_path):
            num_files = sum([len(files) for _, _, files in os.walk(local_ravdess_path)])
            print(f"✓ RAVDESS già presente: {local_ravdess_path}")
            print(f"Numero di file: {num_files}")
            return local_ravdess_path
        
        # 3. Setup Kaggle
        if not setup_kaggle():
            print("\n❌ Impossibile procedere: setup Kaggle fallito.")
            print("Assicurati che il file 'kaggle.json' sia nella cartella corretta.")
            return None
        
        # 4. Scarica con Kaggle CLI (più veloce: download diretto, no cache)
        print("Scaricamento RAVDESS via Kaggle CLI...")
        os.makedirs(local_ravdess_path, exist_ok=True)
        
        cmd = f'kaggle datasets download -d uwrfkaggler/ravdess-emotional-speech-audio -p "{local_ravdess_path}" --unzip --quiet'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Errore Kaggle CLI: {result.stderr}")
            return None
        
        num_files = sum([len(files) for _, _, files in os.walk(local_ravdess_path)])
        print(f"✓ RAVDESS scaricato: {local_ravdess_path}")
        print(f"Numero totale di file: {num_files}")
        return local_ravdess_path
        
    except Exception as e:
        print(f"Errore durante il download di RAVDESS: {e}")
        return None


def dowload_iemocap_local():
    """
    Scarica IEMOCAP localmente nella directory 'data/iemocap/' del progetto.
    Se esiste già, salta il download.
    Usa Kaggle CLI per velocità: scarica direttamente senza cache intermedia.
    
    Returns:
        str: Percorso completo del dataset scaricato se successo
        None: Se fallisce il download o setup Kaggle
    """
    print("\n--- Download IEMOCAP (locale) ---")
    
    try:
        # 1. Definisci il percorso di destinazione
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        local_iemocap_path = os.path.join(project_root, 'data', 'iemocap')
        
        # 2. Verifica se esiste già
        if os.path.exists(local_iemocap_path) and os.listdir(local_iemocap_path):
            num_files = sum([len(files) for _, _, files in os.walk(local_iemocap_path)])
            print(f"✓ IEMOCAP già presente: {local_iemocap_path}")
            print(f"Numero di file: {num_files}")
            # Ritorna il percorso corretto della cartella IEMOCAP_full_release
            iemocap_full_release = os.path.join(local_iemocap_path, 'IEMOCAP_full_release')
            if os.path.exists(iemocap_full_release):
                return iemocap_full_release
            else:
                return local_iemocap_path
        
        # 3. Setup Kaggle
        if not setup_kaggle():
            print("\n❌ Impossibile procedere: setup Kaggle fallito.")
            print("Assicurati che il file 'kaggle.json' sia nella cartella corretta.")
            return None
        
        # 4. Scarica con Kaggle CLI
        print("Scaricamento IEMOCAP via Kaggle CLI...")
        os.makedirs(local_iemocap_path, exist_ok=True)
        
        cmd = f'kaggle datasets download -d dejolilandry/iemocapfullrelease -p "{local_iemocap_path}" --unzip --quiet'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Errore Kaggle CLI: {result.stderr}")
            return None
        
        num_files = sum([len(files) for _, _, files in os.walk(local_iemocap_path)])
        print(f"✓ IEMOCAP scaricato: {local_iemocap_path}")
        print(f"Numero totale di file: {num_files}")
        
        # Ritorna il percorso corretto della cartella IEMOCAP_full_release
        iemocap_full_release = os.path.join(local_iemocap_path, 'IEMOCAP_full_release')
        if os.path.exists(iemocap_full_release):
            return iemocap_full_release
        else:
            return local_iemocap_path
        
    except Exception as e:
        print(f"Errore durante il download di IEMOCAP: {e}")
        return None


if __name__ == "__main__":
    print("="*80)
    print("Download Dataset Locali (Kaggle CLI)")
    print("="*80)
    
    ravdess_path = dowload_ravdess_local()
    iemocap_path = dowload_iemocap_local()
    
    print("\n" + "="*80)
    print("RIEPILOGO DOWNLOAD")
    print("="*80)
    print(f"RAVDESS: {'✅ {ravdess_path}' if ravdess_path else '❌ Fallito'}")
    print(f"IEMOCAP: {'✅ {iemocap_path}' if iemocap_path else '❌ Fallito'}")
    print("="*80)
