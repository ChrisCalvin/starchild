
'''
Expert models for regime-specific forecasting.

This module provides:
- BaseExpert interface (Protocol)
- ARForecaster (simple, immediate use)
- ExpertBankManager (lifecycle management)
- ContinualExpertBank (with EWC support)
- FFG foundation (for future development)
'''

from .base_expert import BaseExpert, ExpertMetadata
from .ar_forecaster import ARForecaster
from .expert_bank import (
    ExpertBankManager,
    ContinualExpertBank,
    create_ar_expert_factory,
    create_expert_bank
)

__all__ = [
    'BaseExpert',
    'ExpertMetadata',
    'ARForecaster',
    'ExpertBankManager',
    'ContinualExpertBank',
    'create_ar_expert_factory',
    'create_expert_bank'
]
