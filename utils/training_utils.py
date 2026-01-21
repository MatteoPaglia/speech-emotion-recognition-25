"""
Training utilities for Speech Emotion Recognition
Contiene classi e funzioni helper per il training (non core training logic)
"""

import torch
from pathlib import Path


class SimpleEarlyStopping:
    """Versione semplice: si ferma appena la validation loss smette di migliorare."""
    def __init__(self, patience=5):
        self.patience = patience
        self.best_loss = None
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            print(f"📊 Best val loss: {val_loss:.4f}")
        elif val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            print(f"✓ Migliorato! New best: {val_loss:.4f}")
        else:
            self.counter += 1
            print(f"⚠️ Nessun miglioramento ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"🛑 STOP! Nessun miglioramento per {self.patience} epoche")


def save_swa_checkpoint(swa_model, path):
    """
    Salva il checkpoint SWA estraendo i pesi dal wrapper AveragedModel.
    Questo garantisce compatibilità per il caricamento futuro (es. Noisy Student).
    """
    unwrapped_state_dict = swa_model.module.state_dict()
    torch.save(unwrapped_state_dict, path)


def update_bn_custom(loader, model, device):
    """
    Wrapper per update_bn che gestisce il formato custom del nostro dataloader.
    Il nostro loader restituisce dizionari {'audio_features': tensor, ...}
    mentre update_bn si aspetta direttamente i tensori.
    """
    model.train()
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum
    
    if not momenta:
        return
    
    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0
    
    for batch in loader:
        # Estrai il tensore audio dal dizionario
        inputs = batch['audio_features'].to(device)
        model(inputs)
    
    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)
