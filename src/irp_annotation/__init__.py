"""IRP (Interest, Rights, Power) annotation module.

This module provides functionality to annotate negotiation conversations with
IRP strategies using GPT-4 with majority voting for quality assurance.
"""

from .irp_annotation import (
    annotate_conversation_with_irp,
    annotate_model_conversations,
    annotate_kodis_conversations,
    IRP_STRATEGY_PROMPT,
    SPLIT_INSTRUCTION,
    ANNOTATION_INSTRUCTION
)

from .merge_annotations import (
    merge_model_annotations,
    merge_kodis_annotations,
    combine_consecutive_utterances
)

__all__ = [
    'annotate_conversation_with_irp',
    'annotate_model_conversations',
    'annotate_kodis_conversations',
    'merge_model_annotations',
    'merge_kodis_annotations',
    'combine_consecutive_utterances',
    'IRP_STRATEGY_PROMPT',
    'SPLIT_INSTRUCTION',
    'ANNOTATION_INSTRUCTION'
]
