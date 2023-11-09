import logging
import torch

from typing import List
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    # input fields: ["audio", "spectrogram", "duration","text", "text_encoded", "audio_path"]
    # output fields: ["spectrogram", "text_encoded", "text_encoded_length", "text"]

    spectrograms = [row['spectrogram'].squeeze(0).T for row in dataset_items]
    audios = [row["audio"].squeeze(0) for row in dataset_items]
    spectrograms_length = [row['spectrogram'].shape[-1] for row in dataset_items]
    texts_encoded = [row['text_encoded'].squeeze(0) for row in dataset_items]
    texts_encoded_length = [row['text_encoded'].shape[-1] for row in dataset_items]
    
    return {
        'spectrogram': pad_sequence(spectrograms, batch_first=True).transpose(1, 2),
        'audio': pad_sequence(audios, batch_first=True),
        'spectrogram_length': torch.tensor(spectrograms_length, dtype=torch.int32),
        'text_encoded': pad_sequence(texts_encoded, batch_first=True),
        'text_encoded_length': torch.tensor(texts_encoded_length, dtype=torch.int32),
        'text': [row['text'] for row in dataset_items],
        'audio_path': dataset_items[0]['audio_path']
    }