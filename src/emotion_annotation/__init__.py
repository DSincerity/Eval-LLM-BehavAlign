"""Emotion annotation module using EmoBERTa.

This module provides emotion classification capabilities for conversation data
using the EmoBERTa transformer model.
"""

from .emotion_classification import (
    load_tokenizer_model,
    inference,
    annotate_model_conversations,
    annotate_kodis_conversations,
    EMOTIONS,
    ID2EMOTION
)

__all__ = [
    'load_tokenizer_model',
    'inference',
    'annotate_model_conversations',
    'annotate_kodis_conversations',
    'EMOTIONS',
    'ID2EMOTION'
]
