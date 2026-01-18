"""
Models package for Speech Emotion Recognition.

Contains different neural network architectures for emotion classification:
- CRNN_BiLSTM: Convolutional Recurrent Neural Network with Bidirectional LSTM
- CRNN_BiGRU: Convolutional Recurrent Neural Network with Bidirectional GRU

Future architectures:
- ResNet-based models for transfer learning
- VGG-based models for transfer learning
- Transformer-based models
"""

from .crnn_bilstm import CRNN_BiLSTM
from .crnn_bigru import CRNN_BiGRU


def get_model(model_name, **kwargs):
    """
    Factory function per istanziare modelli di emotion recognition.
    
    Args:
        model_name (str): Nome del modello da istanziare. Opzioni disponibili:
            - 'crnn_lstm' o 'lstm': CRNN con BiLSTM
            - 'crnn_gru' o 'gru': CRNN con BiGRU
            (Futuro: 'resnet18', 'resnet34', 'resnet50', 'vgg16', 'vgg19')
        **kwargs: Parametri specifici del modello (batch_size, time_steps, dropout, etc.)
    
    Returns:
        nn.Module: Modello istanziato
    
    Raises:
        ValueError: Se il nome del modello non è riconosciuto
    
    Examples:
        >>> # LSTM model
        >>> model = get_model('lstm', batch_size=32, time_steps=200, dropout=0.4)
        
        >>> # GRU model
        >>> model = get_model('gru', batch_size=32, time_steps=200, dropout=0.4)
        
        >>> # Con nome esplicito
        >>> model = get_model('crnn_lstm', batch_size=32, time_steps=200)
    """
    models = {
        'crnn_lstm': CRNN_BiLSTM,
        'lstm': CRNN_BiLSTM,  # Alias per comodità
        'crnn_gru': CRNN_BiGRU,
        'gru': CRNN_BiGRU,  # Alias per comodità
        # Aggiungerai qui i modelli di transfer learning:
        # 'resnet18': ResNetEmotion18,
        # 'resnet34': ResNetEmotion34,
        # 'resnet50': ResNetEmotion50,
        # 'vgg16': VGGEmotion16,
        # 'vgg19': VGGEmotion19,
    }
    
    model_name_lower = model_name.lower()
    if model_name_lower not in models:
        available = ', '.join(sorted(models.keys()))
        raise ValueError(
            f"Modello '{model_name}' non trovato.\n"
            f"Modelli disponibili: {available}"
        )
    
    return models[model_name_lower](**kwargs)


# Export per backward compatibility e import diretti
__all__ = ['CRNN_BiLSTM', 'CRNN_BiGRU', 'get_model']
