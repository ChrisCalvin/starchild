"""
Training script for the Variational Autoencoder (VAE) model.
"""

import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
from sasaie_core.models.vae import VAE
from sasaie_trader.connectors import CSVDataConnector, HummingbotAPIConnector
from sasaie_trader.preprocessing import DataPreprocessor, HummingbotDataPreprocessor
from sasaie_trader.config import HummingbotConfig
from sasaie_trader.hummingbot_manager import HummingbotManager

def vae_loss_function(recon_x, x, mu, logvar):
    """Calculates the VAE loss, which is a sum of reconstruction loss and KL divergence."""
    # Reconstruction loss (e.g., Mean Squared Error)
    recon_loss = torch.nn.functional.mse_loss(recon_x, x, reduction='sum')

    # KL divergence loss
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kld_loss

async def main(args):
    """Main training loop."""
    # 1. Load and Preprocess Data
    print("Loading and preprocessing data...")
    
    hb_config = HummingbotConfig() # Load default config
    manager = HummingbotManager(hb_config)

    if args.connector_type == "csv":
        connector = CSVDataConnector(file_path=args.data_path)
        feature_columns = ['value'] # Example for market_data.csv
        preprocessor = DataPreprocessor(feature_columns=feature_columns)
    elif args.connector_type == "hummingbot_api":
        connector = HummingbotAPIConnector(
            manager=manager,
            connector_name=args.connector_name,
            trading_pair=args.trading_pair,
        )
        feature_columns = ["best_bid_price", "best_ask_price", "mid_price", "spread"] # Example for order book data
        preprocessor = HummingbotDataPreprocessor(feature_columns=feature_columns)
    else:
        raise ValueError(f"Unsupported connector type: {args.connector_type}")

    if args.input_dim is None:
        args.input_dim = len(feature_columns)
        print(f"Input dimension not provided, defaulting to {args.input_dim} based on feature columns.")

    # Load all data into memory to fit the preprocessor
    # For HummingbotAPIConnector, this will stream live data until interrupted or a certain amount is collected
    # For training, we might want to collect a fixed amount or for a duration.
    # For now, we'll collect a fixed number of samples for simplicity.
    all_raw_data = []
    print(f"Connecting to data source using {args.connector_type} connector...")
    await connector.connect()
    try:
        data_stream = connector.stream_data()
        # Collect a fixed number of samples for training
        for i in range(args.num_samples_to_collect):
            try:
                data_point = await data_stream.__anext__()
                all_raw_data.append(data_point)
            except StopAsyncIteration:
                print("Data stream exhausted.")
                break
            except Exception as e:
                print(f"Error collecting data point: {e}")
                break
    finally:
        await connector.disconnect()

    if not all_raw_data:
        print("Error: No data loaded. Please check the data path/connector configuration.")
        return

    # Fit the preprocessor on the entire dataset
    preprocessor.fit(all_raw_data)

    # Process all data points using the fitted preprocessor
    all_features = [preprocessor.process(raw_data_point)['features'] for raw_data_point in all_raw_data]

    # Create a single tensor from the list of features and create a DataLoader
    dataset = TensorDataset(torch.stack(all_features))
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(f"Data loaded and normalized successfully. Number of samples: {len(dataset)}")

    # 2. Initialize Model and Optimizer
    print("Initializing model and optimizer...")
    # The VAE now also needs to be able to reconstruct the normalized data
    model = VAE(input_dim=args.input_dim, latent_dim=args.latent_dim, hidden_dim=args.hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # 3. Training Loop
    print("Starting training...")
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for batch_idx, (data,) in enumerate(data_loader):
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(data)
            loss = vae_loss_function(recon_batch, data, mu, logvar)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loader.dataset)
        print(f'====> Epoch: {epoch+1} Average loss: {avg_loss:.4f}')

    print("Training complete.")

    # 4. Save the trained model (optional)
    if args.save_path:
        print(f"Saving model to {args.save_path}...")
        torch.save(model.state_dict(), args.save_path)
        print("Model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VAE Training Script')
    parser.add_argument('--connector-type', type=str, default='csv', choices=['csv', 'hummingbot_api'], help='Type of data connector to use')
    parser.add_argument('--data-path', type=str, default='data/market_data.csv', help='Path to the training data CSV file (for CSV connector)')
    parser.add_argument('--num-samples-to-collect', type=int, default=100, help='Number of samples to collect from connector for training')
    parser.add_argument('--api-url', type=str, default=None, help='Hummingbot API URL (for hummingbot_api connector)')
    parser.add_argument('--connector-name', type=str, default='binance_paper_trade', help='Hummingbot connector name (for hummingbot_api connector)')
    parser.add_argument('--trading-pair', type=str, default='ETH-USDT', help='Trading pair (for hummingbot_api connector)')
    parser.add_argument('--api-username', type=str, default=None, help='Hummingbot API username (for hummingbot_api connector)')
    parser.add_argument('--api-password', type=str, default=None, help='Hummingbot API password (for hummingbot_api connector)')
    parser.add_argument('--input-dim', type=int, default=5, help='Dimensionality of the input features')
    parser.add_argument('--hidden-dim', type=int, default=64, help='Dimensionality of the hidden layer')
    parser.add_argument('--latent-dim', type=int, default=2, help='Dimensionality of the latent space')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('--save-path', type=str, default='vae_model.pt', help='Path to save the trained model')
    
    args = parser.parse_args()
    asyncio.run(main(args))