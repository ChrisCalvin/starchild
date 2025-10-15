AMAIE Testing Strategy & Conventions

  This document outlines the testing strategy and conventions for the AMAIE project. A
  robust and well-maintained test suite is critical for ensuring the correctness,
  reliability, and performance of the Active Inference trading system. These guidelines
  promote consistency, improve test maintainability, and facilitate effective quality
  assurance.

  ---

  1. Core Principles

   * Automated: All tests should be automated and runnable via a command-line interface.
   * Fast Feedback: Unit tests should be fast to execute, providing rapid feedback to
     developers.
   * Comprehensive Coverage: Strive for high code coverage, especially for critical business
     logic and core AI components.
   * Reproducible: Tests should produce the same results every time they are run, regardless
     of the environment or order of execution.
   * Maintainable: Tests should be easy to read, understand, and update as the codebase
     evolves.
   * Pytest Framework: pytest is the chosen testing framework for its flexibility, powerful
     fixtures, and clear reporting.

  ---

  2. Test Types and Purpose

  AMAIE employs a multi-layered testing approach:

  2.1. Unit Tests

   * Purpose: To test individual, isolated units of code (functions, methods, small
     classes) in isolation from external dependencies.
   * Scope: Focus on the smallest testable parts of the application.
   * Characteristics: Fast, isolated, deterministic.
   * Location: tests/<package_name>/unit/test_*.py
   * Example: Testing a single mathematical function in amaie_core.components.model, or a
     utility function in amaie_core.config.config_utils.

  2.2. Integration Tests

   * Purpose: To test the interaction and communication between multiple integrated units or
     components. This includes testing the data flow between PerceptionEngine, ModelManager,
     and Planner.
   * Scope: Verify that different modules or services work correctly together.
   * Characteristics: May involve mocking external systems (e.g., actual exchanges, complex
     external APIs) but test the internal integration points.
   * Location: tests/<package_name>/integration/test_*.py
   * Example: Testing that PerceptionEngine output is correctly consumed by ModelManager,
     and ModelManager's BeliefState is correctly processed by Planner.

  2.3. End-to-End (E2E) Tests

   * Purpose: To test the entire system or a significant subsystem from end-to-end,
     simulating real-world user scenarios.
   * Scope: Verify the complete flow, often involving multiple services, databases, and
     external integrations (e.g., a full trading cycle from data ingestion to order
     execution).
   * Characteristics: Slower, may require a deployed environment (e.g., Docker Compose
     setup), less isolated, but provide high confidence in overall system functionality.
   * Location: tests/<package_name>/e2e/test_*.py (or tests/system/test_*.py)
   * Example: Running a simulated trading strategy against a mock exchange, verifying that
     orders are placed, positions are managed, and risk limits are respected.

  2.4. Performance Benchmarks

   * Purpose: To measure the performance characteristics (e.g., latency, throughput, memory
     usage) of critical components or end-to-end flows.
   * Scope: Identify performance bottlenecks and ensure that changes do not degrade
     performance.
   * Characteristics: Often use dedicated benchmarking tools (pytest-benchmark), run less
     frequently.
   * Location: tests/<package_name>/benchmark/test_*.py
   * Example: Benchmarking the inference time of the ModelManager or the EFE calculation in
     the Planner.

  ---

  3. Test File and Function Naming

   * Test Files: Must start with test_ or end with _test.py.
       * Example: test_perception_engine.py, my_module_test.py
   * Test Functions/Methods: Must start with test_.
       * Example: def test_process_valid_data():, def test_invalid_input_raises_error():
   * Test Classes: Must start with Test.
       * Example: class TestModelManager:

  ---

  4. Test Structure (Arrange-Act-Assert)

  Each test should clearly follow the Arrange-Act-Assert pattern:

   1. Arrange (Given): Set up the test environment, initialize objects, prepare input data,
      and mock dependencies.
   2. Act (When): Execute the code under test.
   3. Assert (Then): Verify the outcome, checking return values, side effects, or raised
      exceptions.

   * Example:

    1     import pytest
    2     import torch
    3     from amaie_core.api.observation import Observation
    4     from amaie_core.components.perception import PerceptionEngine
    5     from amaie_core.config.perception import PerceptionConfig,
      ScatteringConfig
    6 
    7     # Arrange: Fixtures for setup
    8     @pytest.fixture
    9     def perception_config():
   10         return PerceptionConfig(
   11             feature_names=["market_state_t1"],
   12             scattering=ScatteringConfig(J=8, Q=1, shape=[1, 128])
   13         )
   14 
   15     @pytest.fixture
   16     def perception_engine(perception_config):
   17         return PerceptionEngine(perception_config)
   18 
   19     def test_perception_engine_output_format(perception_engine):
   20         # Arrange: Prepare dummy raw data
   21         raw_data = torch.randn(1, 128)
   22 
   23         # Act: Process the data
   24         observation = perception_engine.process(raw_data)
   25 
   26         # Assert: Verify the output
   27         assert isinstance(observation, Observation)
   28         assert "market_state_t1" in observation.features
   29         assert isinstance(observation.features["market_state_t1"],
      torch.Tensor)
   30         assert observation.features["market_state_t1"].shape == (1, 2)

  ---

  5. Test Data Management

   * Small, Representative Data: Use the smallest possible dataset that adequately covers
     the test case.
   * Mocking: Use unittest.mock or pytest-mock to replace external dependencies (e.g.,
     network calls, database access, complex computations) with controlled mock objects.
     This ensures tests are fast and isolated.
   * Fixtures: Leverage pytest fixtures for reusable setup and teardown logic (e.g.,
     creating temporary files, initializing common objects).
   * Parameterized Tests: Use pytest.mark.parametrize to run the same test logic with
     different input values, reducing code duplication.

   * Example (Mocking):

    1     from unittest.mock import MagicMock
    2     from amaie_core.components.model import ModelManager
    3 
    4     def test_model_manager_loads_model_from_file(mocker):
    5         # Arrange: Mock the ModelParser.load_model method
    6         mock_model_definition = MagicMock()
    7         mock_model_definition.model_name = "MockModel"
    8         mock_model_definition.variables = [] # Minimal definition
    9         mock_model_definition.factors = [] # Minimal definition
   10 
   11         mocker.patch('amaie_core.components.model.ModelParser.load_model',
   12                      return_value=mock_model_definition)
   13         mocker.patch('amaie_core.components.model.ModelParser.validate_model'
      ,
   14                      return_value=True)
   15 
   16         # Act: Initialize ModelManager
   17         model_manager = ModelManager(config=MagicMock(model_file="mock.yaml",
      device="cpu", dtype="float32"))
   18 
   19         # Assert: Verify that load_model was called and manager is 
      initialized
   20 
      amaie_core.components.model.ModelParser.load_model.assert_called_once_with(
   21             mocker.ANY # We don't care about the exact path, just that it was
      called
   22         )
   23         assert model_manager.model_definition.model_name == "MockModel"

  ---

  6. Test Execution

   * Pytest Command: Tests are executed using the pytest command.
   * Markers: Use pytest.mark.<marker_name> to categorize tests (e.g., @pytest.mark.unit,
     @pytest.mark.integration, @pytest.mark.slow). This allows running subsets of tests.
       * To run tests with a specific marker: pytest -m <marker_name>
       * To skip tests with a specific marker: pytest -m "not <marker_name>"
   * Coverage: Run tests with coverage reporting to identify untested code: pytest 
     --cov=amaie_core --cov-report=html
   * CI/CD Integration: Tests should be integrated into the CI/CD pipeline to run
     automatically on every code push.

  ---

  7. Assertions

   * Use assert statements directly.
   * Provide clear and concise assertion messages when the reason for failure might not be
     immediately obvious.
   * Use pytest.raises to test for expected exceptions.

   * Example:

   1     import pytest
   2 
   3     def test_division_by_zero_raises_error():
   4         with pytest.raises(ZeroDivisionError, match="division by zero"):
   5             1 / 0
   6 
   7     def test_list_length_is_correct():
   8         my_list = [1, 2, 3]
   9         assert len(my_list) == 3, "List should contain exactly 3 elements"

  ---

