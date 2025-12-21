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
            print(f"ERRORE: La directory scaricata è vuota o non esiste")
            return False

        # 3. Pulisci la destinazione se esiste già (per evitare conflitti o file vecchi)
        if os.path.exists(destination_dir):
            print(f"La cartella locale '{destination_dir}' esiste già. La rimuovo per aggiornarla...")
            shutil.rmtree(destination_dir)

        if(filter_keyword is None):
            # 4. Copia dalla cache alla cartella del progetto
            print(f"Copia dei file nella cartella di lavoro: {destination_dir}...")
            shutil.copytree(cached_path, destination_dir)
        else :
            # 4. Copia SELETTIVA dalla cache alla cartella del progetto
            print(f"Inizio la copia selettiva (cercando: '{filter_keyword}')...")
            num_files_copied = 0
            
            # Attraversa ricorsivamente la cartella scaricata
            for root, dirs, files in os.walk(cached_path):
                # Calcola il percorso relativo all'interno del dataset
                relative_path = os.path.relpath(root, cached_path)
                
                # Criterio di filtro: Ignora se non è la radice e non contiene la parola chiave
                if filter_keyword and filter_keyword.lower() not in relative_path.lower():
                    continue
                    
                # Calcola il percorso di destinazione
                dest_path = os.path.join(destination_dir, relative_path)
                
                # Se la cartella di destinazione non esiste, la crea
                os.makedirs(dest_path, exist_ok=True)
                
                # Copia i file presenti in questa sottocartella
                for file in files:
                    source_file = os.path.join(root, file)
                    dest_file = os.path.join(dest_path, file)
                    shutil.copy2(source_file, dest_file) 
                    num_files_copied += 1

        # 5. Verifica finale
        num_files = sum([len(files) for _, _, files in os.walk(destination_dir)])
        print(f"{target_folder_name.upper()} pronto in: {destination_dir}")
        print(f"Numero totale di file copiati: {num_files}")
        return True
        
    except Exception as e:
        print(f"Errore durante il download di {target_folder_name}: {e}")
        return False


def download_ravdess():
    return download_dataset_via_hub("uwrfkaggler/ravdess-emotional-speech-audio", "ravdess")


def download_iemocap():
    return download_dataset_via_hub("dejolilandry/iemocapfullrelease", "iemocap", filter_keyword="Impro")


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
        #print(f"IEMOCAP: {'✅ Successo' if iemocap_ok else '❌ Fallito'}")
        print("="*60)

        
        if ravdess_ok:  # and iemocap_ok:
            print("\n🎉 Tutti i dataset sono stati scaricati con successo!")
        else:
            print("\n⚠️  Alcuni download sono falliti. Controlla i messaggi sopra.")
        
    else:
        print("\n❌ Impossibile procedere ai download: setup Kaggle fallito.")
        print("Assicurati che il file 'kaggle.json' sia nella cartella corretta.")