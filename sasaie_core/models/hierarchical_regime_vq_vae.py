# Part of the new RegimeVAE architecture as of 2025-10-13

"""
Hierarchical, Compositional, and Continual VQ-VAE

This file defines the core components for a VQ-VAE world model that is:
1.  Hierarchical: Learns regimes at multiple scales.
2.  Compositional: Learns how regimes at one scale compose into regimes at higher scales.
3.  Continual: Can be updated with new data and new regimes without catastrophically forgetting old ones.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


# ============================================================================
# 1. CORE BUILDING BLOCKS
# ============================================================================

class Encoder(nn.Module):
    """Encodes input features (e.g., a matrix profile) to a latent vector."""
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """Reconstructs input features from a latent vector."""
    def __init__(self, latent_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, z):
        return self.net(z)


class CompositionMatrix(nn.Module):
    """
    Learns the probabilistic grammar of how lower-level patterns (codes) 
    compose into higher-level ones.
    """
    def __init__(self, n_codes_below: int, n_codes_current: int):
        super().__init__()
        self.composition = nn.Parameter(torch.randn(n_codes_current, n_codes_below) * 0.1)
        self.transitions = nn.Parameter(torch.randn(n_codes_current, n_codes_current) * 0.1)
        self.register_buffer('co_occurrence', torch.zeros(n_codes_current, n_codes_below))

    def forward(self, codes_below_dist: torch.Tensor) -> torch.Tensor:
        """Predicts current-level code distribution from lower-level distribution."""
        composition_logits = F.linear(codes_below_dist, self.composition)
        return F.softmax(composition_logits, dim=-1)

    def update_statistics(self, code_current: int, codes_below_dist: torch.Tensor):
        """Update co-occurrence statistics."""
        self.co_occurrence[code_current] += codes_below_dist.squeeze()

    def get_composition_for_code(self, code_idx: int) -> torch.Tensor:
        """Returns the composition vector for a specific higher-level code."""
        return F.softmax(self.composition[code_idx], dim=-1)

    def expand_current_level(self, n_new_codes: int = 1):
        """Adds new rows for new codes discovered at the current level."""
        n_current, n_below = self.composition.shape
        new_composition = nn.Parameter(torch.randn(n_current + n_new_codes, n_below) * 0.1)
        new_composition.data[:n_current] = self.composition.data
        self.composition = new_composition
        
        new_co_occurrence = torch.zeros(n_current + n_new_codes, n_below)
        new_co_occurrence[:n_current] = self.co_occurrence
        self.register_buffer('co_occurrence', new_co_occurrence)

    def expand_below_level(self, n_new_codes: int = 1):
        """Adds new columns for new codes discovered at the level below."""
        n_current, n_below = self.composition.shape
        new_composition = nn.Parameter(torch.randn(n_current, n_below + n_new_codes) * 0.1)
        new_composition.data[:, :n_below] = self.composition.data
        self.composition = new_composition

        new_co_occurrence = torch.zeros(n_current, n_below + n_new_codes)
        new_co_occurrence[:, :n_below] = self.co_occurrence
        self.register_buffer('co_occurrence', new_co_occurrence)


# ============================================================================
# 2. THE CONTINUAL VQ-VAE LAYER (Compositional Unit)
# ============================================================================

class ContinualVQVAELayer(nn.Module):
    """
    A single VQ-VAE layer capable of continual learning (EWC) and dynamic codebook growth.
    This is the fundamental building block of the hierarchical model.
    """
    def __init__(self, input_dim: int, latent_dim: int, initial_codebook_size: int):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.codebook_size = initial_codebook_size

        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)
        self.codebook = nn.Embedding(initial_codebook_size, latent_dim)
        
        self.fisher_matrix = {}
        self.optimal_params = {}
        self.register_buffer('code_usage', torch.zeros(initial_codebook_size))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
        """Standard forward pass for reconstruction and loss calculation."""
        z_e = self.encoder(x)
        z_q, indices, _ = self.quantize(z_e)
        z_q_st = z_e + (z_q - z_e).detach()  # Straight-through estimator
        x_recon = self.decoder(z_q_st)
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        return x_recon, z_q, commitment_loss, indices

    def quantize(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Quantizes a latent vector to the nearest codebook entry."""
        distances = torch.cdist(z_e.unsqueeze(0), self.codebook.weight)
        min_distances, indices = distances.min(dim=-1)
        z_q = self.codebook(indices)
        return z_q.squeeze(0), indices.squeeze(), min_distances.mean().item()

    def encode_and_quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, float]:
        """Encodes and quantizes, returning the quantized vector, index, and distance."""
        z_e = self.encoder(x)
        z_q, index, dist = self.quantize(z_e)
        self.code_usage[index.item()] += 1
        return z_q, index.item(), dist

    def add_new_code(self, novel_pattern_encoding: torch.Tensor) -> int:
        """Dynamically expands the codebook with a new code."""
        with torch.no_grad():
            new_size = self.codebook_size + 1
            new_codebook = nn.Embedding(new_size, self.latent_dim, device=self.codebook.weight.device)
            new_codebook.weight.data[:self.codebook_size] = self.codebook.weight.data
            new_codebook.weight.data[self.codebook_size] = novel_pattern_encoding.squeeze()
            
            self.codebook = new_codebook
            new_idx = self.codebook_size
            self.codebook_size = new_size
            
            self.code_usage = torch.cat([self.code_usage, torch.zeros(1, device=self.code_usage.device)])
            return new_idx

    def compute_fisher(self, data_loader: List[torch.Tensor]):
        """Computes the Fisher Information Matrix for EWC."""
        self.eval()
        fisher = {name: torch.zeros_like(param) for name, param in self.named_parameters() if param.requires_grad}
        
        for data in data_loader:
            self.zero_grad()
            x_recon, _, commitment_loss, _ = self.forward(data)
            loss = F.mse_loss(x_recon, data) + 0.25 * commitment_loss
            loss.backward()
            
            for name, param in self.named_parameters():
                if param.grad is not None and name in fisher:
                    fisher[name] += param.grad.data ** 2
        
        for name in fisher:
            fisher[name] /= len(data_loader)
        
        self.fisher_matrix = fisher
        self.optimal_params = {name: param.clone().detach() for name, param in self.named_parameters()}
        self.train()

    def ewc_loss(self, lambda_ewc: float = 1000.0) -> torch.Tensor:
        """Calculates the Elastic Weight Consolidation loss."""
        if not self.fisher_matrix:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        loss = 0.0
        for name, param in self.named_parameters():
            if name in self.fisher_matrix:
                loss += (self.fisher_matrix[name] * (param - self.optimal_params[name]) ** 2).sum()
        
        return lambda_ewc * loss


# ============================================================================
# 3. HIERARCHICAL ORCHESTRATOR
# ============================================================================

class HierarchicalRegimeVQVAE(nn.Module):
    """
    The orchestrator class. Manages a hierarchy of ContinualVQVAELayers,
    and learns the compositional relationships between them.
    """
    def __init__(self, scales: List[int], input_dim: int, latent_dim: int, codebook_sizes: List[int]):
        super().__init__()
        self.scales = scales
        self.n_levels = len(scales)

        # Create a stack of continual VQ-VAE layers
        self.layers = nn.ModuleList([
            ContinualVQVAELayer(input_dim, latent_dim, size) for size in codebook_sizes
        ])

        # Create composition and attention modules between layers
        self.composition_matrices = nn.ModuleList([
            CompositionMatrix(codebook_sizes[i], codebook_sizes[i+1]) for i in range(self.n_levels - 1)
        ])
        self.cross_attentions = nn.ModuleList([
            nn.MultiheadAttention(latent_dim, num_heads=4, batch_first=True) for _ in range(self.n_levels - 1)
        ])

    def forward(self, multi_scale_features: Dict[int, torch.Tensor]) -> Tuple[Dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Full hierarchical forward pass for end-to-end training.
        Orchestrates forward pass through all ContinualVQVAELayers,
        calculating reconstruction, VQ, and commitment losses.
        """
        reconstructions = {}
        all_vq_loss = torch.tensor(0.0, device=multi_scale_features[self.scales[0]].device)
        all_commitment_loss = torch.tensor(0.0, device=multi_scale_features[self.scales[0]].device)
        
        z_contexts = {} # To pass quantized latents from lower to higher levels

        for level, scale in enumerate(self.scales):
            x = multi_scale_features[scale]
            layer = self.layers[level]

            # Encode the input features for this level
            z_e = layer.encoder(x)

            # Apply cross-attention context from the level below
            if level > 0:
                # Use the quantized latent from the level below as context
                context_z = z_contexts[level - 1]
                
                # Ensure context_z has a batch dimension for MultiheadAttention
                # and match dimensions if necessary
                if context_z.dim() == 2: # (batch_size, latent_dim)
                    context_z = context_z.unsqueeze(1) # (batch_size, 1, latent_dim)
                
                # z_e also needs to be (batch_size, seq_len, embed_dim) for MultiheadAttention
                # Assuming z_e is (batch_size, latent_dim)
                query = z_e.unsqueeze(1) # (batch_size, 1, latent_dim)

                attended_z, _ = self.cross_attentions[level-1](
                    query, context_z, context_z
                )
                z_e = z_e + attended_z.squeeze(1) # Residual connection, squeeze back to (batch_size, latent_dim)

            # Quantize the context-aware encoding
            # layer.forward returns x_recon, z_q, commitment_loss, indices
            x_recon, z_q, commitment_loss_layer, _ = layer.forward(x) # Pass original x for reconstruction target
            
            # VQ loss (codebook loss) - encourage codebook embeddings to move towards encoder outputs
            # This is (z_q - z_e.detach())**2
            vq_loss_layer = F.mse_loss(z_q, z_e.detach())

            reconstructions[scale] = x_recon
            all_vq_loss += vq_loss_layer
            all_commitment_loss += commitment_loss_layer
            
            z_contexts[level] = z_q # Store quantized latent for next level's context
                
        return reconstructions, all_vq_loss, all_commitment_loss

    def hierarchical_encode(self, mp_features: Dict[int, torch.Tensor]) -> Dict[int, Tuple[int, float]]:
        """
        Encodes multi-scale features into a hierarchical set of regime codes,
        passing context from lower to higher levels.
        """
        codes = {}
        z_contexts = {}
        
        for level, scale in enumerate(self.scales):
            x = mp_features[scale]
            layer = self.layers[level]
            
            # Encode the input features for this level
            z_e = layer.encoder(x)
            
            # Apply cross-attention context from the level below
            if level > 0 and (level - 1) in z_contexts:
                context_z = z_contexts[level - 1]
                attended_z, _ = self.cross_attentions[level-1](
                    z_e.unsqueeze(0), context_z.unsqueeze(0), context_z.unsqueeze(0)
                )
                z_e = z_e + attended_z.squeeze(0) # Residual connection

            # Quantize the context-aware encoding
            z_q, code_tensor, distance = layer.quantize(z_e)
            code = code_tensor.item()
            
            codes[scale] = (code, distance)
            z_contexts[level] = z_q
            
            # Update composition statistics
            if level > 0:
                prev_code = codes[self.scales[level-1]][0]
                prev_layer = self.layers[level-1]
                codes_below_dist = F.one_hot(torch.tensor(prev_code), num_classes=prev_layer.codebook_size).float()
                self.composition_matrices[level-1].update_statistics(code, codes_below_dist)
                
        return codes

    def explain_composition(self, code: int, level: int) -> Dict:
        """Explains what lower-level patterns compose into a given code."""
        if level == 0:
            return {"message": "Bottom level has no composition."}
        
        composition_vec = self.composition_matrices[level-1].get_composition_for_code(code)
        top_k = min(5, len(composition_vec))
        values, indices = torch.topk(composition_vec, top_k)
        
        return {
            "level": level, "code": code, "scale": self.scales[level],
            "composed_from": [
                {"lower_code": idx.item(), "lower_scale": self.scales[level-1], "contribution": val.item()}
                for idx, val in zip(indices, values)
            ]
        }