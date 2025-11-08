"""Evaluation metrics for L2L negotiation analysis.

This module contains various metrics for analyzing LLM-to-LLM negotiations:
- Anger trajectory analysis (DTW distance and AUC)
- Strategic behavior analysis (IRP framework with JSD)
- Linguistic gap analysis (LIWC features)
- Linguistic entrainment analysis (nCLiD)
"""

from .anger_trajectory import analyze_anger_trajectory
from .strategic_behavior import analyze_strategic_behavior, STRATEGY_ORDER
from .linguistic_gap import analyze_linguistic_gap
from .linguistic_entrainment import analyze_linguistic_entrainment

__all__ = [
    'analyze_anger_trajectory',
    'analyze_strategic_behavior',
    'analyze_linguistic_gap',
    'analyze_linguistic_entrainment',
    'STRATEGY_ORDER',
]
