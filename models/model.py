from torch import nn
import torch

class CRNN_BiLSTM(nn.Module):
    def __init__(self, batch_size, time_steps, mel_band=128, channel=1):
        super().__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(channel, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        )
        
        self.block4 = nn.Sequential(
             nn.Conv2d(512, 1024, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
             nn.BatchNorm2d(1024),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        )

        # Calcolo dimensione feature per la LSTM:
        # Dopo 3 MaxPool (2x2), l'altezza (frequenza) diventa 128 / 8 = 16.
        # I canali sono diventati 64.
        # Quindi ogni step temporale avrà un vettore di: 64 * 16 = 1024 feature.
        self.lstm_input_size = 64 * 16
        self.hidden_size = 128
        self.num_classes = 4  # Ho solo 4 classi di emozioni (Neutral, Happy, Sad, Angry)

        # Projection Layer per ridurre gradualmente le dimensioni (evita collo di bottiglia)
        # Riduciamo da 1024 -> 128 con una densa prima della LSTM
        self.projection = nn.Linear(self.lstm_input_size, self.hidden_size)

        # Bi-LSTM
        self.lstm = nn.LSTM(
            input_size=self.hidden_size,  # Input dal projection layer (128, non 1024)
            hidden_size=self.hidden_size, 
            num_layers=1,             # Solitamente 1 o 2 strati
            bidirectional=True, 
            batch_first=True          # Importante: input sarà (Batch, Time, Features)
        )

        # Layer per l'Attention Mechanism
        # Proietta l'output della LSTM (256 feature perché bidirezionale) in uno score scalare
        self.attention_linear = nn.Linear(self.hidden_size * 2, 1)

        # Layer di classificazione finale
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(self.hidden_size * 2, self.num_classes)




    def forward(self, x):
        # 1. Passaggio attraverso i blocchi CNN
        # Input: (Batch, 1, 128, Time)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        #x = self.block4(x)
        
        # A questo punto x ha dimensioni: (Batch, 64, 16, Time_Ridotto)
        
        # 2. Permutazione delle dimensioni
        # La LSTM vuole il Tempo come seconda dimensione.
        # Spostiamo le dimensioni: (Batch, Time_Ridotto, 64, 16)
        x = x.permute(0, 3, 1, 2)
        
        # 3. LA RESHAPE VERA E PROPRIA
        # Fondiamo le ultime due dimensioni (Canali e Frequenza) in una sola (Feature)
        # -1 dice a PyTorch: "Calcola tu questa dimensione (che sarà 64*16 = 1024)"
        x = x.reshape(x.size(0), x.size(1), -1)
        
        # Ora x ha dimensioni: (Batch, Time_Ridotto, 1024) ed è pronto per il projection
        
        # 3.5. Passaggio nel projection layer + attivazione
        # Riduciamo da 1024 -> 128 gradualmente per evitare collo di bottiglia
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
        
        