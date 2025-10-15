"""
End-to-end training script for the hierarchical perception model, including
the Normalizing Flow hierarchy, the BeliefUpdater, and the GAT-based synthesizer.
"""

import argparse
import torch
from torch.optim import Adam
import itertools

from sasaie_core.config.loader import GenerativeModelConfigLoader
from sasaie_core.models.multi_scale_nf import MultiScaleNFModel
from sasaie_core.components.belief_updater import BeliefUpdater
from sasaie_core.components.global_belief_fusion import BeliefSynthesizerGAT

def main(args):
    # 1. Load Configuration
    print("Loading hierarchical model configuration...")
    config_loader = GenerativeModelConfigLoader(args.config_path)
    model_config = config_loader.config['model']
    gnn_config = config_loader.config['gnn_synthesizer']
    scale_configs = {scale['name']: scale for scale in model_config['scales']}

    # 2. Initialize Components
    print("Initializing models...")
    # Initialize the multi-scale NF model
    nf_model = MultiScaleNFModel(scale_configs)
    
    # Initialize the orchestrator
    belief_updater = BeliefUpdater(nf_model, config_loader)

    # Initialize the GAT-based synthesizer
    scale_dims = {name: conf['latent_dim'] for name, conf in scale_configs.items()}
    belief_synthesizer = BeliefSynthesizerGAT(
        scale_dims=scale_dims,
        output_dim=gnn_config['output_dim'],
        heads=gnn_config['attention_heads']
    )

    # 3. Setup Optimizer
    # Combine parameters from all models for joint optimization
    all_params = itertools.chain(nf_model.parameters(), belief_synthesizer.parameters())
    optimizer = Adam(all_params, lr=args.learning_rate)

    # 4. Training Loop (with dummy data for demonstration)
    print("Starting training loop (using dummy data)...")
    nf_model.train()
    belief_synthesizer.train()

    for epoch in range(args.epochs):
        # --- Dummy Data Generation for one time step ---
        # In a real scenario, this would come from a DataLoader streaming real data.
        previous_states = {name: torch.randn(1, conf['latent_dim']) for name, conf in scale_configs.items()}
        perception_features = {name: torch.randn(1, 2) for name in scale_configs.keys()} # Dummy perception features
        # --- End of Dummy Data ---

        optimizer.zero_grad()

        # a. Run the hierarchical belief update
        current_states = belief_updater.update(previous_states, perception_features)

        # b. Run the GAT synthesizer
        consolidated_belief = belief_synthesizer(current_states)

        # c. Calculate Loss
        # The primary loss is the negative log-likelihood of the underlying NFs.
        # We need to re-calculate the log_prob for the states that were sampled.
        # This is a simplification; a more rigorous loss would use the full model evidence (ELBO).
        total_nll = 0
        for scale_name, state_t in current_states.items():
            # This part is complex as we need the context used to generate the sample.
            # For simplicity, we'll just use a placeholder loss.
            # A full implementation requires passing context through or re-building it.
            pass
        
        # Placeholder loss: Encourage the output variance to be high
        loss = -consolidated_belief.var()

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch: {epoch+1}/{args.epochs}, Placeholder Loss: {loss.item():.4f}')

    print("Training complete.")

    # 5. Save Models
    if args.save_path:
        print(f"Saving models to {args.save_path}...")
        torch.save(nf_model.state_dict(), f"{args.save_path}_nf.pt")
        torch.save(belief_synthesizer.state_dict(), f"{args.save_path}_gat.pt")
        print("Models saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hierarchical Perception Model Training Script')
    parser.add_argument('--config-path', type=str, default='configs/generative_model.yaml', help='Path to the model config file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--save-path', type=str, default='hierarchical_model', help='Base path to save the trained models')
    
    args = parser.parse_args()
    main(args)
