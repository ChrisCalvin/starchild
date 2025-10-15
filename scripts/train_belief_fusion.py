"""
Training script for the Global Belief Fusion pipeline (GNN + VAE).
"""

import argparse
import torch
from torch.optim import Adam
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Data
from typing import List

# Import all necessary components
from sasaie_core.components.belief_aggregator import (
    BeliefVectorizer,
    BeliefGraphConstructor,
    BeliefAggregatorGNN
)
from sasaie_core.components.global_belief_fusion import GlobalBeliefFusionVAE
from sasaie_core.components.belief_updater import BFEBeliefUpdater
from sasaie_core.models.nf import ConditionalSplineFlow

# --- VAE Loss Function --- #
def vae_loss_function(recon_x, x, mu, logvar):
    # Use Mean Squared Error for reconstruction loss for continuous data
    MSE = F.mse_loss(recon_x, x.view(-1, x.shape[-1]), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD

# --- Dummy Dataset Creation --- #
def create_dummy_belief_graphs(num_graphs: int, num_nodes_per_graph: int, context_dim: int) -> List[List[torch.Tensor]]:
    """Creates a dummy dataset of belief graphs (represented by their observation contexts)."""
    dataset = []
    for _ in range(num_graphs):
        graph_contexts = [torch.randn(1, context_dim) for _ in range(num_nodes_per_graph)]
        dataset.append(graph_contexts)
    return dataset


def main(args):
    # --- 1. Initialize all components --- #
    print("Initializing belief fusion components...")
    
    # We need a dummy NF model for the BFE updater
    nf_model = ConditionalSplineFlow(latent_dim=args.latent_dim, context_dim=args.context_dim)
    
    # The BFE updater is used by the vectorizer
    bfe_updater = BFEBeliefUpdater(
        nf_model=nf_model, 
        latent_dim=args.latent_dim, 
        context_dim=args.context_dim
    )
    
    # The vectorizer is used by the graph constructor
    belief_vectorizer = BeliefVectorizer(belief_updater=bfe_updater)
    
    # The graph constructor creates the input for the GNN
    belief_graph_constructor = BeliefGraphConstructor(belief_vectorizer=belief_vectorizer)
    
    # The GNN aggregates beliefs
    gnn_model = BeliefAggregatorGNN(
        in_channels=args.latent_dim, # Input to GNN is the latent dim from BFE updater
        hidden_channels=args.gnn_hidden_dim,
        out_channels=args.gnn_out_dim
    )
    
    # The VAE produces the ConsolidatedBeliefState
    fusion_vae = GlobalBeliefFusionVAE(
        input_dim=args.gnn_out_dim, # Input to VAE is the output of GNN
        latent_dim=args.cbs_dim, # Latent space of VAE is the ConsolidatedBeliefState
        hidden_dim=args.vae_hidden_dim
    )

    # --- 2. Create Dummy Data --- #
    print("Creating dummy dataset...")
    dummy_data = create_dummy_belief_graphs(args.num_graphs, args.num_nodes, args.context_dim)
    
    # --- 3. Setup Optimizer --- #
    # Combine parameters of both GNN and VAE for joint optimization
    params = list(gnn_model.parameters()) + list(fusion_vae.parameters())
    optimizer = Adam(params, lr=args.learning_rate)

    # --- 4. Training Loop --- #
    print("Starting belief fusion training...")
    gnn_model.train()
    fusion_vae.train()
    
    for epoch in range(args.epochs):
        total_loss = 0
        for graph_contexts in dummy_data:
            optimizer.zero_grad()
            
            # a. Construct the belief graph
            belief_graph = belief_graph_constructor.construct_graph(graph_contexts)
            
            # b. Pass through GNN
            aggregated_belief = gnn_model(belief_graph)
            
            # c. Pass through VAE
            recon_aggregated_belief, mu, logvar = fusion_vae(aggregated_belief)
            
            # d. Calculate loss
            loss = vae_loss_function(recon_aggregated_belief, aggregated_belief, mu, logvar)
            
            # e. Backpropagate
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dummy_data)
        print(f'====> Epoch: {epoch+1} Average Loss: {avg_loss:.4f}')

    print("Belief fusion training complete.")

    # --- 5. Save Models --- #
    if args.save_path:
        print(f"Saving models to {args.save_path}...")
        torch.save(gnn_model.state_dict(), f"{args.save_path}_gnn.pt")
        torch.save(fusion_vae.state_dict(), f"{args.save_path}_vae.pt")
        print("Models saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Belief Fusion Training Script')
    # Model dimensions
    parser.add_argument('--latent-dim', type=int, default=2, help='Latent dimension of the base NFs/BFE updater')
    parser.add_argument('--context-dim', type=int, default=10, help='Context dimension for the BFE updater')
    parser.add_argument('--gnn-hidden-dim', type=int, default=32, help='Hidden dimension for the GNN')
    parser.add_argument('--gnn-out-dim', type=int, default=64, help='Output dimension of the GNN')
    parser.add_argument('--vae-hidden-dim', type=int, default=128, help='Hidden dimension for the Fusion VAE')
    parser.add_argument('--cbs-dim', type=int, default=32, help='Dimensionality of the ConsolidatedBeliefState')
    # Training parameters
    parser.add_argument('--num-graphs', type=int, default=100, help='Number of dummy graphs to generate for training')
    parser.add_argument('--num-nodes', type=int, default=3, help='Number of nodes (beliefs) per graph')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('--save-path', type=str, default='belief_fusion', help='Path prefix to save the trained models')
    
    args = parser.parse_args()
    main(args)
