from typing import List, Dict, Tuple
from collections import Counter
import torch

from sasaie_core.evolution.registry import GraphComponentRegistry
from sasaie_core.evolution.morphisms import RGMorphism

class HierarchyProcessor:
    """
    Analyzes successful policy compositions to identify recurring patterns and
    formalizes them as new, more abstract RG-Morphisms.
    This is the core of the structural learning loop.
    """

    def __init__(self, registry: GraphComponentRegistry):
        """
        Initializes the HierarchyProcessor.

        Args:
            registry: The registry where new morphisms will be registered.
        """
        self.registry = registry

    def analyze_and_propose_new_morphisms(self, successful_policies: List[List[RGMorphism]]):
        """
        The main method for structural learning. Analyzes a history of successful
        policies and proposes new morphisms.

        Args:
            successful_policies: A list of policies, where each policy is a list
                                 of RGMorphism objects.
        """
        print("Analyzing successful policies for structural learning...")

        # Step 1: Find frequent/successful sequences
        # This implementation focuses on frequent 2-grams (sequences of 2 morphisms)
        # A more advanced implementation would use proper frequent sequence mining
        # and incorporate EFE reduction analysis.
        print("  Identifying candidate sequences (2-grams)...")
        candidate_sequence_morphisms = self._find_frequent_sequence(successful_policies, n_gram=2, min_frequency=2)

        if not candidate_sequence_morphisms:
            print("No new candidate morphisms found.")
            return

        # Step 2: Train a new morphism for the identified sequence
        print(f"  Training new morphism for sequence: {[m.name for m in candidate_sequence_morphisms]}")
        new_morphism = self._train_new_morphism(candidate_sequence_morphisms)

        # Step 3: Register the new morphism
        print(f"  Registering new morphism: '{new_morphism.name}'")
        self.registry.register_morphism(new_morphism)

        print("Structural learning cycle complete.")

    def _find_frequent_sequence(self, policies: List[List[RGMorphism]], n_gram: int = 2, min_frequency: int = 2) -> List[RGMorphism]:
        """
        Identifies frequent n-gram sequences of RGMorphisms within policies.
        Returns the most frequent sequence (list of RGMorphism objects) found.
        """
        sequence_counts = Counter()
        all_morphism_sequences = []

        for policy in policies:
            morphism_names = tuple(m.name for m in policy) # Use names for counting
            all_morphism_sequences.append(morphism_names)

            # Generate n-grams
            for i in range(len(morphism_names) - n_gram + 1):
                sub_sequence = morphism_names[i : i + n_gram]
                sequence_counts[sub_sequence] += 1

        most_common_sequence_names = None
        max_frequency = 0

        for seq_names, count in sequence_counts.items():
            if count >= min_frequency and count > max_frequency:
                most_common_sequence_names = seq_names
                max_frequency = count
        
        if most_common_sequence_names:
            # Reconstruct the RGMorphism objects for the most common sequence
            # This assumes the original RGMorphism objects are available or can be retrieved
            # from the registry if needed. For simplicity, we'll try to find them in the first policy
            # where the sequence appears.
            for policy in policies:
                policy_names = tuple(m.name for m in policy)
                if most_common_sequence_names in [policy_names[i : i + n_gram] for i in range(len(policy_names) - n_gram + 1)]:
                    start_index = [policy_names[i : i + n_gram] for i in range(len(policy_names) - n_gram + 1)].index(most_common_sequence_names)
                    return policy[start_index : start_index + n_gram]
        return []

    def _train_new_morphism(self, sequence: List[RGMorphism]) -> RGMorphism:
        """
        Creates a new RGMorphism that encapsulates the given sequence.
        Infers IO and context dimensions from the sequence.
        Includes conceptual training paths for end-to-end and conditional matching.
        """
        if not sequence:
            raise ValueError("Sequence of morphisms cannot be empty.")

        # Infer IO and context dimensions from the first and last morphisms in the sequence
        input_io_dim = sequence[0].io_dim
        output_io_dim = sequence[-1].io_dim
        
        # Assuming context_dim remains consistent or is taken from the first morphism
        context_dim = sequence[0].context_dim 

        new_name = "_".join([m.name for m in sequence])
        print(f"    Creating new RGMorphism named '{new_name}' with input_io_dim={input_io_dim}, output_io_dim={output_io_dim}, context_dim={context_dim}...")
        
        # For now, use dummy values for num_layers and modality.
        # The actual architecture (num_layers) and modality of the new composite morphism
        # would need to be determined more intelligently or be part of a meta-learning process.
        new_morphism = RGMorphism(
            name=new_name,
            io_dim=input_io_dim,
            context_dim=context_dim,
            modality="composite", # Indicate it's a composite morphism
            num_layers=2 # Placeholder: actual layers might depend on sequence complexity
        )

        # --- Conceptual Training Path 1: End-to-End Matching ---
        print("    Conceptual Training Path 1: End-to-End Matching...")
        # Goal: Train new_morphism to map input of sequence[0] to output of sequence[-1]
        # 1. Data Generation:
        #    - Generate a dataset of (input_z, input_context) pairs.
        #    - For each pair, pass input_z through the entire 'sequence' of RGMorphisms
        #      to get the final output_z_target.
        #    - Store (input_z, input_context, output_z_target) triples.
        # 2. Training Loop:
        #    - Initialize an optimizer (e.g., Adam) for new_morphism.
        #    - For each epoch:
        #        - Sample a batch of (input_z, input_context, output_z_target).
        #        - Pass input_z through new_morphism to get predicted_output_z.
        #        - Calculate loss (e.g., KL divergence between predicted_output_z's distribution
        #          and output_z_target's distribution, or MSE if using point estimates).
        #        - Perform backpropagation and update new_morphism's parameters.
        print("    ...Placeholder for actual training logic for end-to-end matching...")

        # --- Conceptual Training Path 2: Conditional/Recurrent Matching ---
        print("    Conceptual Training Path 2: Conditional/Recurrent Matching...")
        # Goal: Train new_morphism to approximate the sequence's behavior by leveraging
        #       intermediate conditions or contexts.
        # 1. Data Generation:
        #    - Generate a dataset of (input_z, input_context) pairs.
        #    - For each pair, pass input_z through the 'sequence' step-by-step.
        #    - At each step, record (intermediate_input_z, intermediate_context, intermediate_output_z_target).
        #    - The 'intermediate_context' for new_morphism could be a concatenation of
        #      contexts from the original sequence, or a learned summary.
        # 2. Training Loop:
        #    - Similar to Path 1, but the new_morphism might be trained to predict
        #      the next step's output given the current input and a richer context.
        #    - This could involve a more complex loss function that considers the
        #      fidelity of intermediate transformations.
        print("    ...Placeholder for actual training logic for conditional/recurrent matching...")
        
        return new_morphism
