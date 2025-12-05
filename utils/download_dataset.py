import os
import shutil

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


def download_ravdess():
    """Scarica il dataset RAVDESS"""
    return download_dataset_via_hub("uwrfkaggler/ravdess-emotional-speech-audio", "ravdess")


def download_iemocap():
    """
    Scarica il dataset IEMOCAP usando il dataset che funziona.
    Questo dataset NON dà errore 403.
    """
    print("\n⚠️  NOTA: Sto usando il dataset 'dejolilandry/iemocapfullrelease'")
    print("    che è verificato funzionante e non dà errore 403.\n")
    
    return download_dataset_via_hub("dejolilandry/iemocapfullrelease", "iemocap")


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