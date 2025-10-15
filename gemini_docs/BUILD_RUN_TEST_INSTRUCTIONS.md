**Instructions for Building, Running, and Testing AMAIE 2.0 (AINF)**

This document provides high-level instructions for setting up, operating, and verifying the AMAIE 2.0 project. Given the project's modular and advanced nature, these instructions serve as a guiding framework.

### 1. Prerequisites

Before you begin, ensure you have the following installed:

*   **Python 3.9+:** Recommended for compatibility with modern deep learning libraries.
*   **Git:** For cloning the project repository.
*   **CUDA (for GPU acceleration):** Highly recommended for training and inference. Ensure your GPU drivers are up-to-date and compatible with your PyTorch version.
*   **`conda` or `venv`:** For managing Python virtual environments.

### 2. Setup

1.  **Clone the Repository:**
    ```bash
git clone <repository_url>
cd amaie2.0
    ```
2.  **Create and Activate a Virtual Environment:**
    Using `conda` (recommended):
    ```bash
conda create -n amaie2.0 python=3.9
conda activate amaie2.0
    ```
    Using `venv`:
    ```bash
python -m venv venv
source venv/bin/activate
    ```
3.  **Install Dependencies:**
    Navigate to the project root and install the required packages. It's recommended to install PyTorch with CUDA support first, then other dependencies.
    ```bash
# Example for PyTorch with CUDA 11.8 (adjust as per your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other core dependencies
pip install -r requirements.txt
    ```
    *(Note: `requirements.txt` will be generated based on the external packages discussed, e.g., nflows, PyG, Pyro, Pandas, NumPy.)*

### 3. Building (Conceptual)

AMAIE 2.0 is primarily a Python-based deep learning project. "Building" involves setting up the environment and training the various neural network components. The project's modular structure (e.g., `perception/`, `state_space/`, `belief_fusion/`, `planning/`) means components can often be developed and tested independently before integration.

### 4. Running

Running AMAIE 2.0 typically involves two main phases: **Training** and **Inference/Deployment**.

1.  **Training the Model:**
    *   **Configuration:** Training runs are controlled via configuration files (e.g., YAML files managed by Hydra or OmegaConf). These files define model architectures, hyperparameters, data paths, and training schedules.
    *   **Data Preparation:** Ensure your multi-modal, multi-rate domain data is prepared and accessible as specified in the configuration.
    *   **Execution:** Training scripts will typically be located in a `scripts/` directory or within the respective module directories.
        ```bash
# Example: Train the Order Book VAE
python scripts/train_order_book_vae.py --config-name default_ob_vae

# Example: Train the full AINF system (end-to-end)
python scripts/train_ainf_system.py --config-name full_system_config
        ```
    *   **Monitoring:** Use tools like TensorBoard or Weights & Biases for monitoring training progress, losses, and metrics.

2.  **Running Inference / Deployment:**
    *   **Model Loading:** Trained models (weights) are loaded from checkpoints.
    *   **Configuration:** Inference runs also use configuration files to specify loaded models, data sources, and operational parameters.
    *   **Execution:**
        ```bash
# Example: Run the AINF system in simulation/backtest mode
python main.py --mode backtest --config-name backtest_scenario

# Example: Run the AINF system in real-time (connecting to data streams)
python main.py --mode real_time --config-name live_deployment
        ```
    *   **Real-time Integration:** For live deployment, the system will connect to domain-specific data streams via APIs and execute actions through domain-specific execution engines.

### 5. Testing

Comprehensive testing is crucial for a project of this complexity.

1.  **Unit Tests:**
    *   **Purpose:** Verify the correctness of individual functions, classes, and small modules (e.g., a single NF block, a VAE encoder/decoder, a GNN layer).
    *   **Execution:**
        ```bash
pytest tests/unit/
        ```
2.  **Integration Tests:**
    *   **Purpose:** Verify that different components work correctly when integrated (e.g., Perception Layer feeding into State-Space Model, Belief Fusion working with Planning).
    *   **Execution:**
        ```bash
pytest tests/integration/
        ```
3.  **Performance Tests:**
    *   **Purpose:** Measure latency, throughput, and resource utilization of critical paths (e.g., inference time for the NF state-space model, EFE calculation speed).
    *   **Execution:** Custom scripts will be developed for benchmarking.
        ```bash
python scripts/benchmark_inference.py
        ```

### 6. Development Guidelines (Brief)

*   **Modularity:** Adhere strictly to the defined modular structure. Each module should have clear responsibilities and well-defined APIs.
*   **API-First Design:** Ensure `ainf-core` and `domain-specific-modules` interact exclusively via robust APIs.
*   **Documentation:** Maintain clear and concise inline code documentation and module-level READMEs.
*   **Code Style:** Follow established Python style guides (e.g., PEP 8) and use linters (e.g., Black, Ruff).