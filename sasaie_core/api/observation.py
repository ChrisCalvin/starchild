from dataclasses import dataclass, field
from datetime import datetime
import torch
from typing import Dict, Any

@dataclass
class Observation:
    """Represents a standardized, domain-agnostic input to the world model."""
    features: Dict[str, torch.Tensor]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)