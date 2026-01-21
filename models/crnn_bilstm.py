from torch import nn
import torch

class CRNN_BiLSTM(nn.Module):
    def __init__(self, batch_size, time_steps, mel_band=128, channel=1, dropout=0.4):
        super().__init__()
        self.dropout_rate = dropout
        
        # === VECCHIA IMPLEMENTAZIONE (COMMENTATA) ===
        # self.block1 = nn.Sequential(
        #     nn.Conv2d(channel, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        # )
        # self.block2 = nn.Sequential(
        #     nn.Conv2d(128, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        # )
        # self.block3 = nn.Sequential(
        #     nn.Conv2d(256, 512, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        # )
        # self.block4 = nn.Sequential(  # ⚠️ PROBLEMATICO: 1024 canali sono eccessivi per 1440 campioni
        #      nn.Conv2d(512, 1024, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
        #      nn.BatchNorm2d(1024),
        #      nn.ReLU(),
        #      nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        # )
        
        # === NUOVA IMPLEMENTAZIONE (OTTIMIZZATA) ===
        # Riduzione delle dimensioni: 128 -> 128 -> 256 -> 256 (anzichè -> 512 -> 1024)
        # Aggiunta di Dropout tra i blocchi per ridurre overfitting
        
        self.block1 = nn.Sequential(
            nn.Conv2d(channel, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),  # Dropout dopo il primo blocco
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1)),  # 128 -> 128 (meno canali)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),  # Dropout dopo il secondo blocco
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1)),  # 128 -> 256
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),  # Dropout dopo il terzo blocco
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        )
        
        # Block4 ridotto: 256 -> 128 per ulteriore riduzione parametri
        # Meno memoria = Migliore generalizzazione su Test (small dataset)
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1)),  # 256 -> 128 (ridotto)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),  # Dropout dopo il quarto blocco
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        )

        # Calcolo dimensione feature per la LSTM:
        # Dopo 4 MaxPool (2x2), l'altezza (frequenza) diventa 128 / 16 = 8.
        # I canali sono diventati 128 (ridotto).
        # Quindi ogni step temporale avrà un vettore di: 128 * 8 = 1024 feature
        self.lstm_input_size = 128 * 8  # 1024 (ridotto da 2048)
        self.hidden_size = 128
        self.num_classes = 4  # 4 classi di emozioni (Neutral, Happy, Sad, Angry)

        # Projection Layer per ridurre gradualmente le dimensioni
        # 1024 -> 128 con una densa prima della LSTM
        self.projection = nn.Linear(self.lstm_input_size, self.hidden_size)

        # Bi-LSTM
        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size, 
            num_layers=1,
            bidirectional=True, 
            batch_first=True
        )

        # Layer per l'Attention Mechanism
        self.attention_linear = nn.Linear(self.hidden_size * 2, 1)

        # Layer di classificazione finale con Dropout ridotto (0.3 vs 0.5)
        # Con weight_decay=0.001, possiamo permetterci dropout più basso
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.hidden_size * 2, self.num_classes)




    def forward(self, x):
        # 1. Passaggio attraverso i blocchi CNN
        # Input: (Batch, 1, 128, Time)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        # A questo punto x ha dimensioni: (Batch, 1024, 8, Time_Ridotto)
        
        # 2. Permutazione delle dimensioni
        # La LSTM vuole il Tempo come seconda dimensione.
        # Spostiamo le dimensioni: (Batch, Time_Ridotto, 1024, 8)
        x = x.permute(0, 3, 1, 2)
        
        # 3. LA RESHAPE VERA E PROPRIA
        # Fondiamo le ultime due dimensioni (Canali e Frequenza) in una sola (Feature)
        # -1 dice a PyTorch: "Calcola tu questa dimensione (che sarà 1024*8 = 8192)"
        x = x.reshape(x.size(0), x.size(1), -1)
        
        # Ora x ha dimensioni: (Batch, Time_Ridotto, 8192) ed è pronto per il projection
        
        # 3.5. Passaggio nel projection layer + attivazione
        # Riduciamo da 8192 -> 128 gradualmente per evitare collo di bottiglia
        x = self.projection(x)
        x = torch.relu(x)  # Non-linearità per la rappresentazione
        # Ora x è (Batch, Time_Ridotto, 128), molto più digeribile per la LSTM
        
        # 4. Passaggio attraverso la Bi-LSTM
        # La LSTM restituisce l'output e gli stati hidden (che qui non usiamo, quindi _)
        x, _ = self.lstm(x)
        
        # 5. Attention Mechanism (Calcolo dei pesi)
        # x shape: (Batch, Time, 256) -> perché 128*2 (bidirezionale)
        attention_weights = torch.softmax(self.attention_linear(x), dim=1)
        
        # Moltiplicazione dei pesi per l'output della LSTM (Context Vector)
        x = torch.sum(attention_weights * x, dim=1)
        
        # 6. Classificazione
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x
