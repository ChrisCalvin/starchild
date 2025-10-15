# Part of the new RegimeVAE architecture as of 2025-10-13

from enum import Enum

class ActionType(Enum):
    """Types of actions the agent can take"""
    EXPLOIT = "exploit"  # Use current regime knowledge
    EXPLORE = "explore"  # Gather info about regime
    PREPARE = "prepare"  # Anticipate regime change
    ADAPT = "adapt"     # Respond to novel regime
