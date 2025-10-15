"""
Defines the GraphComponentRegistry for storing and retrieving RG-Morphisms.
"""

from typing import Dict, Optional, List

from sasaie_core.evolution.morphisms import RGMorphism

class GraphComponentRegistry:
    """
    A central registry to store and manage all available RG-Morphisms (skills).
    """

    def __init__(self):
        self._morphisms: Dict[str, RGMorphism] = {}
        self._morphisms_by_modality: Dict[str, List[RGMorphism]] = {}

    def register_morphism(self, morphism: RGMorphism):
        """
        Registers a new morphism.

        Args:
            morphism: The RGMorphism instance to register.
        """
        if morphism.name in self._morphisms:
            raise ValueError(f"Morphism with name '{morphism.name}' is already registered.")
        print(f"Registering morphism: '{morphism.name}'")
        self._morphisms[morphism.name] = morphism
        
        # Register by modality
        self._morphisms_by_modality.setdefault(morphism.modality, []).append(morphism)

    def get_morphism(self, name: str) -> Optional[RGMorphism]:
        """
        Retrieves a morphism by its name.

        Args:
            name: The name of the morphism to retrieve.

        Returns:
            The RGMorphism instance, or None if not found.
        """
        return self._morphisms.get(name)

    def list_morphisms(self) -> List[str]:
        """
        Returns a list of the names of all registered morphisms.
        """
        return list(self._morphisms.keys())

    def get_morphisms_by_modality(self, modality: str) -> List[RGMorphism]:
        """
        Retrieves all morphisms associated with a specific modality.

        Args:
            modality: The modality to filter by.

        Returns:
            A list of RGMorphism instances for the given modality, or an empty list if none found.
        """
        return self._morphisms_by_modality.get(modality, [])
