"""
Defines the BeliefUpdater, which orchestrates the hierarchical NF model.
"""

from typing import Dict, Any, List
import torch
import torch.nn as nn

from sasaie_core.config.loader import GenerativeModelConfigLoader
from sasaie_core.models.multi_scale_nf import MultiScaleNFModel
from sasaie_core.models.nf import ConditionalSplineFlow
from sasaie_core.components.gating import GatingMechanism

class BeliefUpdater(nn.Module):
    """
    Orchestrates the update of beliefs for a single time step using the
    hierarchical, bidirectional state-space model with gating.
    """

    def __init__(self, model: MultiScaleNFModel, config_loader: GenerativeModelConfigLoader):
        """
        Initializes the BeliefUpdater.

        Args:
            model: The MultiScaleNFModel containing all the NF scales.
            config_loader: The loaded configuration providing the execution plan.
        """
        super().__init__()
        self.model = model
        self.config_loader = config_loader
        self.execution_plan = self.config_loader.execution_plan
        self.gating_mechanisms = nn.ModuleDict()

        # Create a mapping from scale name to its corresponding scattering j_index
        self.s1_j_mapping: Dict[str, int] = {}
        for scale_config in self.config_loader.scales:
            if 'j_index' in scale_config:
                self.s1_j_mapping[scale_config['name']] = scale_config['j_index']

    def update(self, previous_states: Dict[str, torch.Tensor], processed_features: Dict[str, Any]) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Executes one full update cycle for a single time step.

        Args:
            previous_states: A dictionary mapping scale names to their latent state
                             tensors from the previous time step (t-1).
            processed_features: A dictionary containing the output from the relevant
                                preprocessor (e.g., ScatteringPreprocessor).

        Returns:
            A tuple containing two dictionaries:
            - The newly computed latent states for the current time step (t).
            - The context vectors used to generate each new state.
        """
        current_states = {}
        current_contexts = {}

        for scale_name in self.execution_plan:
            scale_config = self.config_loader.get_scale_config(scale_name)
            context_tensors: List[torch.Tensor] = []

            # 1. Add own previous state (z_{t-1})
            context_tensors.append(previous_states[scale_name])

            # 2. Add perception features from config
            self._add_perception_features(scale_name, scale_config, processed_features, context_tensors)

            # 3. Add gated couplings from other scales
            self._add_gated_couplings(scale_name, scale_config, processed_features, previous_states, current_states, context_tensors)

            # Assemble the full context vector
            full_context = torch.cat(context_tensors, dim=-1)
            current_contexts[scale_name] = full_context

            # Lazily initialize the main NF model for this scale
            if scale_name not in self.model.nfs:
                self._lazy_init_nf(scale_name, scale_config, full_context)

            # Generate the new state by sampling from the conditional flow
            nf_model_for_scale = self.model.nfs[scale_name]
            new_state = nf_model_for_scale.sample(1, context=full_context).squeeze(1)
            current_states[scale_name] = new_state
            
        return current_states, current_contexts

    def _add_perception_features(self, scale_name, scale_config, processed_features, context_tensors):
        for feature_source in scale_config.get('feature_sources', []):
            if feature_source == 's1_scattering':
                j_index = self.s1_j_mapping.get(scale_name)
                if j_index is None:
                    raise ValueError(f"Scale '{scale_name}' requests s1_scattering but has no j_index mapping.")
                feature = processed_features['s1_features'].get(j_index)
                # If feature is not present for this specific data point, just skip it.
                if feature is not None:
                    context_tensors.append(feature)
            elif feature_source == 's0_scattering':
                context_tensors.append(processed_features['s0_features'])
            elif feature_source == 'vae_latent':
                context_tensors.append(processed_features['vae_latent'])

    def _add_gated_couplings(self, scale_name, scale_config, processed_features, previous_states, current_states, context_tensors):
        for coupling in scale_config.get('couplings', []):
            from_scale = coupling['from_scale']
            coupling_type = coupling['type']

            if 'bottom_up' in coupling_type:
                tensor_to_gate = current_states[from_scale]
            elif 'top_down' in coupling_type:
                tensor_to_gate = previous_states[from_scale]
            else:
                continue # Or raise error for unknown type

            if 's2_gated' in coupling_type:
                j1 = self.s1_j_mapping[from_scale]
                j2 = self.s1_j_mapping[scale_name]
                gating_key = tuple(sorted((j1, j2)))
                gating_signal = processed_features['s2_features'].get(gating_key)
                
                # If the specific S2 feature for this pair is missing, skip the gating.
                # Add the ungated tensor as a fallback.
                if gating_signal is None:
                    context_tensors.append(tensor_to_gate)
                    continue

                # Lazily initialize the gating mechanism
                gate_name = f"{from_scale}_to_{scale_name}"
                if gate_name not in self.gating_mechanisms:
                    self._lazy_init_gate(gate_name, gating_signal, tensor_to_gate)
                
                gating_module = self.gating_mechanisms[gate_name]
                gated_tensor = gating_module(tensor_to_gate, gating_signal)
                context_tensors.append(gated_tensor)
            else: # For simple, non-gated couplings
                context_tensors.append(tensor_to_gate)

    def _lazy_init_nf(self, scale_name, scale_config, full_context):
        latent_dim = scale_config.get('latent_dim', 2)
        context_dim = full_context.shape[-1]
        self.model.nfs[scale_name] = ConditionalSplineFlow(
            latent_dim=latent_dim,
            context_dim=context_dim,
            num_layers=5
        )

    def _lazy_init_gate(self, gate_name, gating_signal, tensor_to_gate):
        signal_dim = gating_signal.shape[-1]
        tensor_dim = tensor_to_gate.shape[-1]
        self.gating_mechanisms[gate_name] = GatingMechanism(signal_dim, tensor_dim)