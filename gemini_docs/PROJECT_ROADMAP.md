# Project Roadmap: AMAIE 2.0 (AINF)

This document outlines the phased development approach for AMAIE 2.0, a Neural Implementation of Hierarchical, Scale-Free Active Inference (AINF). It breaks down the ambitious vision into manageable milestones, facilitating project management, progress tracking, and resource allocation.

## Overall Project Phases

The development of AMAIE 2.0 will proceed in four main phases, each building upon the successful completion of the previous one.

### Phase 1: Foundational Components & Data Pipelines (Months 1-3)

**Goal:** Establish robust data ingestion, initial perception capabilities, and the basic NF state-space model.

*   **1.1 Data Connectors & Preprocessing:**
    *   [x] Implement domain-specific Data Connectors for raw data ingestion (e.g., high-frequency market data, order book).
    *   [x] Develop robust data preprocessing pipelines (time-alignment, cleaning, handling asynchronous/multi-rate streams).
*   **1.2 Perception Layer - Initial Components:**
    *   [c] Implement and train the Order Book VAE for latent representation learning. (c=component complete, pending full validation)
        *   [x] Implement VAE architecture.
        *   [x] Develop training script for VAE.
        *   [c] Train and validate VAE model. (c=initial training complete)
    *   (Optional) Implement and integrate the KAN for adaptive feature selection.
    *   [x] Advanced Scattering Feature Engineering (Volatility Ratios, Lagged Differences).
*   **1.3 Basic NF State-Space Model:**
    *   [x] Implement the core Recursive Conditional Neural Spline Flow.
    *   [x] Develop the BFE minimization module for local belief updates (NF prior + LSTM likelihood).
    *   [x] Train the NF state-space model with initial multi-modal conditioning (Order Book VAE latents, LSTM forecasts).
*   **1.4 Core Infrastructure:**
    *   [x] Set up initial project structure (`sasaie_core/`, `sasaie_trader/`).
    *   [x] Establish basic logging, monitoring, and configuration management.
    *   [x] Implement initial unit tests for core components.

**Deliverables:**
*   Functional data ingestion pipelines.
*   Trained Order Book VAE and integrated LSTM.
*   Working prototype of the recursive NF state-space model.
*   Basic project structure and development environment.

### Phase 2: Hierarchical Belief Fusion & Planning Foundation (Months 4-6)

**Goal:** Implement the global belief fusion mechanisms and the core Forney Factor Graph for planning.

*   **2.1 Hierarchical NF & VAE Bridges:**
    *   [x] Implement parallel NFs for different modalities/scales (e.g., separate NFs for price, order book, news).
    *   [x] Develop and integrate VAE bridges for fusing beliefs across hierarchical levels (radial coupling) and between neighboring modalities (circumferential coupling).
    *   [x] Implement cross-modal consistency scoring for belief fusion.
*   **2.2 Global Belief Fusion:**
    *   [x] Implement the GNN for aggregating beliefs from diverse NFs into a fixed-size representation.
    *   [x] Develop and train the Global Belief Fusion VAE/NF to learn the `ConsolidatedBeliefState`.
*   **2.3 Forney Factor Graph Implementation:**
    *   [x] Set up the basic Forney Factor Graph structure for representing the generative model of future states and actions.
    *   [x] Implement core VMP algorithms for **multi-objective** EFE calculation within the Forney graph.
    *   [x] Define and integrate multiple goal functions (e.g., profit, risk, information gain).
*   **2.4 Initial Policy Selection:**
    *   [x] Implement a basic Policy Selection VAE/NF that takes the `ConsolidatedBeliefState` and proposes simple candidate policies.

**Deliverables:**
*   Functional hierarchical belief fusion system.
*   Working `ConsolidatedBeliefState` generation.
*   Basic Forney Factor Graph capable of EFE calculation for predefined policies.
*   Initial policy proposal mechanism.

### Phase 3: Structural Learning & Compositional Evolution (Months 7-9)

**Goal:** Enable the system to learn and evolve its own optimal hierarchical structure and policy components.

*   **3.1 RG-Morphisms as Micro NFs:**
    *   [x] Formalize RG-Morphisms as evolvable "micro NFs" capable of performing scale transformations.
    *   [x] Integrate these micro NFs into the `GraphComponentRegistry`.
*   **3.2 Dynamic Graph Composition:**
    *   [x] Enhance the `GraphComposer` to dynamically instantiate the Forney Factor Graph by chaining RG-Morphisms and goal groups.
    *   Implement structured typing for type-safe composition.
*   **3.3 Hierarchy Processor & Structural Learning:**
    *   [x] Develop the `HierarchyProcessor` to analyze GNN patterns and formalize new RG-Morphism candidates.
    *   [x] Implement evolutionary algorithms for optimizing RG-Morphisms and their composition based on EFE feedback.
*   **3.4 Policy Evolution Refinement:**
    *   [x] Refine the Policy Selection VAE/NF to propose policies that are compositions of RG-Morphisms, **incorporating multimodal belief consistency and multi-objective goal prioritization.**
    *   [x] Implement context-aware goal weighting and conflict resolution mechanisms.

**Deliverables:**
*   Functional RG-Morphisms (micro NFs).
*   Dynamic Forney Factor Graph composition.
*   Working `HierarchyProcessor` capable of structural learning.
*   System demonstrating basic policy evolution.

### Phase 4: Advanced Features, Optimization & Deployment (Months 10-12+)

**Goal:** Integrate advanced features, optimize for real-time performance, and prepare for deployment.

*   **4.1 Advanced Coupling:**
    *   [x] Implement chordal coupling for non-neighboring NFs, **including explicit cross-modal attention mechanisms.**
        *   [x] Refactored the hierarchical NF coupling to use a physics-informed S2-gated mechanism, implementing learned precision-weighting between scales. (Completed Sept 2025)
    *   [x] Develop modal-specific morphism specialization for enhanced selection.
*   **4.2 Dynamic Pruning:**
    *   [x] Implement dynamic pruning of high-EFE policy traces within the Forney graph.
*   **4.3 Robustness & Error Handling:**
    *   Develop robust error handling for asynchronous data streams and model failures.
    *   Implement mechanisms for handling missing/corrupted data.
*   **4.4 Performance Optimization:**
    *   Apply model distillation, quantization, and hardware-aware design.
    *   Conduct extensive performance benchmarking.
*   **4.5 Deployment & Monitoring:**
    *   Develop tools and procedures for real-time deployment and continuous monitoring.
    *   Establish comprehensive logging and alerting systems.

**Deliverables:**
*   Fully integrated and optimized AINF system.
*   Comprehensive test suite and performance benchmarks.
*   Deployment-ready codebase.
*   Operational monitoring framework.

### Phase X: Domain-Specific Integration (e.g., Trading)

**Goal:** Integrate the core AINF engine with domain-specific modules for real-world application.

*   **X.1 Data Ingestion Integration:**
    *   [x] Integrate `sasaie_trader/connectors.py` with the Perception Layer's data ingestion pipeline.
        *   (Note: Integration will leverage `hummingbot-api` and `hummingbot-api-client` for data acquisition.)
*   **X.2 Domain-Specific Preprocessing Integration:**
    *   [x] Integrate `sasaie_trader/preprocessing.py` into the Perception Layer's feature extraction pipeline.
    *   [x] Validate domain-specific feature generation.
*   **X.3 Action Execution Integration:**
    *   [c] Integrate `sasaie_trader/execution.py` with the Planning Layer's action execution interface. (c=scaffold complete)
        *   [x] Implemented functional EFE-based planner, replacing mock logic. (c=core logic complete, uses mock world model)
    *   [x] Implement and test order routing and trade execution logic. (Requires running Hummingbot API for full integration testing)
*   **X.4 Domain-Specific Model Training & Validation:**
    *   [ ] Train and validate core AINF models (NFs, VAEs, GNN) using real-world trading data.
    *   [ ] Conduct backtesting and simulation for performance evaluation.

**Deliverables:**
*   Fully functional SASAIE system integrated with a specific domain (e.g., trading).
*   Demonstrable performance in a simulated or real-world environment.

### Phase 5: Production Orchestration & Trading Integration (Year 2+)

**Goal:** Implement a robust, flexible orchestration system to manage a containerized Hummingbot instance and integrate it with the SASAIE core for live trading.

*   **5.1 Implement Production Launcher:**
    *   [ ] Create `sasaie_core/launcher` module for orchestration code.
    *   [ ] Add `launch.py` CLI script for system management (`start`, `stop`, `status`).
*   **5.2 Implement Docker Orchestration:**
    *   [ ] Adapt `thor`'s `DockerManager` into `sasaie_core/launcher/docker_manager.py`.
    *   [ ] Create `docker-compose.yml` to define the containerized Hummingbot service.
*   **5.3 Implement System Orchestrator:**
    *   [ ] Adapt `thor`'s `AMAIEOrchestrator` into `sasaie_core/launcher/orchestrator.py`.
    *   [ ] Integrate the orchestrator with the `launch.py` CLI.
*   **5.4 Integrate `sasaie_trader` with Orchestrator:**
    *   [ ] Develop `sasaie_trader.HummingbotConnector` using MQTT (`hbotrc`) to communicate with the containerized bot.
    *   [ ] Connect the `SASAIEOrchestrator` to the `sasaie_trader` to pass actions for execution.
    *   [ ] Integrate data stream from Hummingbot back into the `PerceptionEngine`.

### Phase 6: Physics-Informed & Advanced Modeling (Year 2+)

**Goal:** Integrate deeper physics-informed principles and explore state-of-the-art modeling techniques.

*   **6.1 Physics-Informed State-Space Models:**
    *   [ ] Explore and implement Continuous Normalizing Flows (Neural ODEs) for the core state-space model, informed by principles from the `physics` submodule.
    *   [ ] Investigate and implement adaptive gating mechanisms for Neural ODEs based on lagged scattering coefficient differences to reduce computational load, focusing on differentiable approaches.
    *   [ ] Investigate and implement Chebyshev polynomial fitting for scattering coefficients as input to Neural ODEs, exploring KANs for amortized fitting.
    *   [ ] Investigate and integrate Branched Schrödinger Bridge Matching for modeling and forecasting branched stochastic processes in market dynamics. [Paper](https://arxiv.org/html/2506.09007v1)
*   **6.2 Advanced Structural Learning:**
    *   [x] Implement more sophisticated evolutionary algorithms for optimizing RG-Morphisms, including crossover and more complex mutation operators.
*   **6.3 PINN Integration:**
    *   [ ] Research and develop PINN models for specific market dynamics (e.g., volatility, order flow).
    *   [ ] Integrate PINN models as components in the generative model.