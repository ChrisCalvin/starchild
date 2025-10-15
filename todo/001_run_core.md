1. Understanding the Goal

  The primary objective is to refactor the application's core logic into a centralized, testable, and modular MainPipeline class. This will decouple the data ingress point
   (MarketDataConsumer) from the complex data processing sequence. The final architecture will have the run_core.py script act as a dependency injector, instantiating all
  core components (VQ-VAE, Planner), composing them into a MainPipeline instance, and passing that single pipeline object to the MarketDataConsumer.

  2. Investigation & Analysis

  Before implementing the refactor, a thorough investigation is required to ensure all dependencies and data contracts are understood. The following steps must be taken:

   1. Review Core Component Interfaces:
       * Read sasaie_core/models/hierarchical_regime_vq_vae.py: Confirm the constructor signature for HierarchicalRegimeVQVAE and the input/output signature of its
         hierarchical_encode method.
       * Read sasaie_core/components/planner.py: Confirm the constructor signature for RegimeAwarePlanner and the input/output signatures of its update_beliefs and
         select_best_policy methods.
       * Read sasaie_trader/preprocessing.py: Confirm the constructor signature for HierarchicalStreamingMP and the input/output of its update method.

   2. Analyze Configuration:
       * Read configs/generative_model.yaml: This file is the source of truth for model configuration. I must identify all parameters needed to initialize the components
         (e.g., scales, codebook_sizes, input_dim, latent_dim). The plan must include a mechanism for loading this YAML file in run_core.py.

   3. Examine the Data Ingress Point:
       * Read sasaie_trader/mqtt_consumer.py: Confirm the exact logic within the _on_message callback, specifically how the msg.payload is handled.
       * Read run_core.py: Confirm how the MarketDataConsumer is currently instantiated.

   4. Answer Critical Questions:
       * Data Flow: What is the precise, step-by-step sequence of method calls that will occur within the MainPipeline.process() method? What data object is passed from one
         step to the next?
       * Initial State: The HierarchicalStreamingMP requires an initial time series buffer before it can be used. How will the pipeline manage this "warm-up" period?
       * Dependencies: What will be the complete list of dependencies for the MainPipeline class? This includes not just the VQ-VAE and Planner, but also configuration
         parameters like scales, dimensions, and buffer sizes.
       * Expert Bank: The RegimeAwarePlanner requires an expert_bank. What is the strategy for creating and providing this dependency, given that the actual forecasting
         experts (Factor Graphs) are not yet implemented?
       * Error Handling: What is the strategy for handling a failure in one of the pipeline stages?

  3. Proposed Strategic Approach

  The refactoring will be executed in three distinct phases to ensure a controlled and logical transition.

  Phase 1: Create the `MainPipeline` Component

  A new file, sasaie_core/pipeline.py, will be created. It will contain the MainPipeline class.

   * `__init__` method:
       * It will accept all necessary components and configurations via dependency injection. Its signature will be: __init__(self, vqvae_model, planner, scales, 
         mp_input_dim, initial_buffer_size).
       * It will initialize internal state, including a ts_buffer (deque) and a placeholder for the mp_stream object, which will be None initially.
   * `process(self, raw_payload: bytes)` method:
       * This method will contain the entire step-by-step processing logic, formerly planned for the MarketDataConsumer.
       1. Decode and validate the raw JSON payload.
       2. Extract the price and append it to the internal ts_buffer.
       3. Handle the warm-up period: If the mp_stream is not yet initialized, check if the buffer is full. If so, instantiate HierarchicalStreamingMP.
       4. If the stream is active, call mp_stream.update() to get the latest matrix profiles.
       5. Preprocess the matrix profiles into fixed-size tensors.
       6. Pass the features to vqvae_model.hierarchical_encode() to get regime codes.
       7. Pass the regime codes to planner.update_beliefs() to get a structured belief object.
       8. Placeholder: Generate mock forecasts, as the real "Expert Bank" is not yet implemented.
       9. Pass the beliefs and forecasts to planner.select_best_policy() to get the optimal policy.
       10. Log the chosen policy and its rationale for verification.

  Phase 2: Decouple the `MarketDataConsumer`

  The sasaie_trader/mqtt_consumer.py file will be refactored.

   * `__init__` method: The signature will be simplified to __init__(self, mqtt_host, mqtt_port, topic, pipeline: MainPipeline). It will no longer have any knowledge of the
     VQ-VAE or other core components.
   * `_on_message` method: The logic will be reduced to a single line: self.pipeline.process(msg.payload). This fully decouples the data source from the processing logic.

  Phase 3: Update the Application Entry Point (`run_core.py`)

  This script will become the central "assembler" for the application.

   1. It will be modified to import all necessary classes: MainPipeline, HierarchicalRegimeVQVAE, RegimeAwarePlanner, and MarketDataConsumer.
   2. It will load the configuration from configs/generative_model.yaml.
   3. It will use this configuration to instantiate the core components:
       * HierarchicalRegimeVQVAE
       * A mock expert_bank (e.g., an empty defaultdict)
       * RegimeAwarePlanner
   4. It will then instantiate the MainPipeline, injecting the newly created components and configuration values.
   5. Finally, it will instantiate the MarketDataConsumer, injecting the MainPipeline instance, and start the consumer loop.

  4. Verification Strategy

  The success of this refactor will be verified without relying on full end-to-end behavior.

   1. Unit Testing:
       * A new test file, tests/sasaie_core/unit/test_pipeline.py, must be created.
       * The MainPipeline class will be tested by providing it with mocked versions of the VQ-VAE and Planner. The tests will verify that the process method calls the
         components' methods in the correct sequence and handles the initial buffering state correctly.
       * The unit tests for MarketDataConsumer will be updated to verify that it correctly calls the process method on the mocked pipeline object it receives.
   2. Integration Testing:
       * A new integration test will be created that instantiates the full pipeline with real components (as run_core.py does).
       * The test will manually call the pipeline.process() method with a sequence of mock MQTT payloads.
       * Assertions will focus on verifying that the data flows through the entire pipeline without errors and that the final output (the selected Policy object) is correctly
         formed. It will not validate the correctness of the policy, only the integrity of the pipeline.
   3. Logging: The MainPipeline.process method must be instrumented with detailed INFO and DEBUG logs for each stage. A successful run of the application will be verified by
      observing the log output and confirming that data flows through each step as expected.

  5. Anticipated Challenges & Considerations

   * Configuration Loading: A robust mechanism for loading the YAML configuration and using it to construct the objects in run_core.py is required. This may necessitate a
     simple config-parsing utility.
   * Initial State Management: The logic for filling the initial time series buffer before the HierarchicalStreamingMP can start must be handled carefully to avoid errors on
      startup.
   * Mocking Dependencies: The RegimeAwarePlanner's dependency on an expert_bank for generating forecasts must be addressed. The strategic plan is to use a mock or
     placeholder for now, with the understanding that implementing the real expert bank is a subsequent task.
   * Data Format Assumption: The plan assumes a simple JSON payload from MQTT. This contract must be formally documented and tested. Any deviation will break the pipeline at
      the first step.
