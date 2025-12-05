import os
import shutil
import kagglehub

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
        print(f"❌ ERRORE: File 'kaggle.json' non trovato in: {json_source}")
        print("Assicurati di averlo trascinato nella cartella 'utils'!")
        return False

    os.makedirs(target_dir, exist_ok=True)

    try:
        shutil.copy(json_source, target_file)
        os.chmod(target_file, 0o600)  # Permessi richiesti da Kaggle
        print("✅ Kaggle configurato con successo.")
        return True
    except Exception as e:
        print(f"❌ Errore configurazione Kaggle: {e}")
        return False


def download_dataset_via_hub(dataset_slug, target_folder_name):
    """
    Funzione generica che scarica un dataset tramite kagglehub e lo sposta
    nella cartella di progetto desiderata.
    """
    print(f"\n--- Download {target_folder_name.upper()} ---")
    
    destination_dir = f"./{target_folder_name}"
    
    try:
        # 1. Scarica (o recupera dalla cache)
        print(f"Contatto KaggleHub per scaricare: {dataset_slug}...")
        cached_path = kagglehub.dataset_download(dataset_slug)
        print(f"✓ Dataset scaricato nella cache di sistema: {cached_path}")

        # 2. Verifica che i file esistano
        if not os.path.exists(cached_path) or not os.listdir(cached_path):
            print(f"❌ ERRORE: La directory scaricata è vuota o non esiste")
            return False

        # 3. Pulisci la destinazione se esiste già (per evitare conflitti o file vecchi)
        if os.path.exists(destination_dir):
            print(f"La cartella locale '{destination_dir}' esiste già. La rimuovo per aggiornarla...")
            shutil.rmtree(destination_dir)

        # 4. Copia dalla cache alla cartella del progetto
        print(f"Copia dei file nella cartella di lavoro: {destination_dir}...")
        shutil.copytree(cached_path, destination_dir)
        
        # 5. Verifica finale
        num_files = sum([len(files) for _, _, files in os.walk(destination_dir)])
        print(f"✅ {target_folder_name.upper()} pronto in: {destination_dir}")
        print(f"✅ Numero totale di file copiati: {num_files}")
        return True
        
    except Exception as e:
        print(f"❌ Errore durante il download di {target_folder_name}: {e}")
        return False


def download_dataset_via_hub_selective(dataset_slug, target_folder_name, filter_keyword=None):
    """
    Scarica un dataset e copia selettivamente solo le cartelle/file
    che contengono 'filter_keyword' (case-insensitive) nel nome del percorso.
    """
    print(f"\n--- Download {target_folder_name.upper()} ---")
    
    destination_dir = f"./{target_folder_name}"
    
    try:
        # 1. Scarica (o recupera dalla cache)
        print(f"Contatto KaggleHub per scaricare: {dataset_slug}...")
        cached_path = kagglehub.dataset_download(dataset_slug)
        print(f"✓ Dataset scaricato nella cache di sistema: {cached_path}")

        # ... (Omissione punti 2 e 3 per brevità, sono gli stessi) ...

        # Pulisci la destinazione se esiste già
        if os.path.exists(destination_dir):
            print(f"La cartella locale '{destination_dir}' esiste già. La rimuovo per aggiornarla...")
            shutil.rmtree(destination_dir)
        os.makedirs(destination_dir, exist_ok=True) # Crea la cartella principale

        # 4. Copia SELETTIVA dalla cache alla cartella del progetto
        print(f"Inizio la copia selettiva (cercando: '{filter_keyword}')...")
        num_files_copied = 0
        
        # Attraversa ricorsivamente la cartella scaricata
        for root, dirs, files in os.walk(cached_path):
            # Calcola il percorso relativo all'interno del dataset
            relative_path = os.path.relpath(root, cached_path)
            
            # Criterio di filtro: Ignora se non è la radice e non contiene la parola chiave
            if filter_keyword and filter_keyword.lower() not in relative_path.lower():
                # Esempio: salta la cartella /content/iemocap/Session1/dialog/transcriptions/Ses01F_Script01
                # ma non salta Ses01F_Impro01
                continue
                
            # Calcola il percorso di destinazione
            dest_path = os.path.join(destination_dir, relative_path)
            
            # Se la cartella di destinazione non esiste, la crea
            os.makedirs(dest_path, exist_ok=True)
            
            # Copia i file presenti in questa sottocartella
            for file in files:
                # Puoi aggiungere un filtro anche sui tipi di file se necessario (es. solo .wav)
                source_file = os.path.join(root, file)
                dest_file = os.path.join(dest_path, file)
                shutil.copy2(source_file, dest_file) # copy2 copia anche i metadati
                num_files_copied += 1

        # 5. Verifica finale
        print(f"✅ {target_folder_name.upper()} pronto in: {destination_dir}")
        print(f"✅ Numero totale di file copiati: {num_files_copied}")
        return True
        
    except Exception as e:
        print(f"❌ Errore durante il download selettivo di {target_folder_name}: {e}")
        return False
    
def download_ravdess():
    """Scarica il dataset RAVDESS"""
    return download_dataset_via_hub("uwrfkaggler/ravdess-emotional-speech-audio", "ravdess")


def download_iemocap():
    """
    Scarica solo i file di improvvisazione dal dataset IEMOCAP.
    """
    print("\n⚠️  NOTA: Sto scaricando solo i dati di 'improvvisazione' (Impro).")
    return download_dataset_via_hub_selective(
        "dejolilandry/iemocapfullrelease", 
        "iemocap", 
        filter_keyword="Impro" # <-- Filtro applicato qui
    )


if __name__ == "__main__":
    print("="*60)
    print("DOWNLOAD AUTOMATICO DATASET")
    print("="*60)
    
    # Esegui il setup e poi i download
    if setup_kaggle():
        print("\n" + "="*60)
        
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