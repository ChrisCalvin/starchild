AMAIE Coding Standards & Style Guide

  This document outlines the coding standards and style guidelines for the AMAIE project.
  Adherence to these guidelines ensures code consistency, readability, maintainability, and
  facilitates collaborative development. While automated tools like black handle formatting,
  this guide focuses on higher-level structural and semantic conventions.

  ---

  1. Core Principles

   * Readability: Code should be easy to understand for anyone reading it, including future
     self.
   * Consistency: All code should look and behave as if written by a single person.
   * Clarity over Brevity: Prefer clear, explicit code over overly clever or concise code
     that sacrifices understanding.
   * Maintainability: Code should be easy to modify, extend, and debug.
   * Pythonic: Embrace Python's idiomatic features and best practices (PEP 8, PEP 20 - The
     Zen of Python).

  ---

  2. Naming Conventions

  Follow PEP 8 guidelines, with specific AMAIE project adaptations:

   * Modules (`.py` files): lowercase_with_underscores (snake\_case).
       * Example: amaie_core/components/perception.py
   * Packages (directories): lowercase_with_underscores.
       * Example: amaie_core/api/
   * Classes: CamelCase (CapWords).
       * Example: PerceptionEngine, BeliefState
   * Functions & Methods: lowercase_with_underscores.
       * Example: process_raw_data(), _calculate_uncertainty()
   * Variables: lowercase_with_underscores.
       * Example: market_state_t1, expected_free_energy
   * Constants (global, module-level): ALL_CAPS_WITH_UNDERSCORES.
       * Example: MAX_RETRIES, DEFAULT_TIMEOUT
   * Private (Internal Use) Members: Prefix with a single underscore (_).
       * Example: _private_method(), _internal_variable
   * Type Variables (for generics): CamelCase with _T suffix.
       * Example: _T_Observation, _T_ModelConfig

  ---

  3. Type Hinting

  All new code and significant modifications to existing code must use type hints (PEP 484).

   * Purpose: Improves code readability, enables static analysis (e.g., mypy, pyright), and
     helps catch errors early.
   * Syntax:
       * Use standard library typing module for complex types (List, Dict, Optional, Union,
         Any, Callable, Tuple, TypeVar).
       * Use built-in types for simple cases (str, int, float, bool).
       * For Python 3.9+, use built-in generic types (list[str], dict[str, Any]).
   * Consistency: Apply type hints consistently across function signatures, variable
     assignments, and class attributes.
   * Example:

    1     from typing import Dict, Any, List, Optional
    2     import torch
    3     from datetime import datetime
    4 
    5     class MyAgentComponent:
    6         def __init__(self, name: str, config: Dict[str, Any]):
    7             self.name: str = name
    8             self._config: Dict[str, Any] = config
    9             self._internal_state: Optional[torch.Tensor] = None
   10 
   11         def process_data(self, data: torch.Tensor, metadata: Dict[str, Any])
      -> List[float]:
   12             """
   13             Processes input data and returns a list of processed values.
   14             """
   15             if self._internal_state is None:
   16                 self._internal_state = torch.zeros_like(data)
   17             processed_values: List[float] = (data + self
      ._internal_state).tolist()
   18             return processed_values
   19 
   20         def _update_state(self, new_value: float) -> None:
   21             # Private method
   22             if self._internal_state is not None:
   23                 self._internal_state += new_value

  ---

  4. Docstrings

  All modules, classes, public functions, and public methods must have docstrings (PEP 257).
  Use the Google style for docstrings.

   * Module Docstrings: Briefly describe the module's purpose.
   * Class Docstrings: Describe the class's purpose, its main responsibilities, and key
     attributes.
   * Function/Method Docstrings:
       * Brief summary of what the function does.
       * Args: section: List each argument, its type, and a brief description.
       * Returns: section: Describe the return value and its type.
       * Raises: section: Document any exceptions that can be raised.
       * Example: (Optional but highly recommended for complex functions).

   * Example:

    1     """
    2     This module provides the core PerceptionEngine for processing raw market
      data.
    3     """
    4 
    5     import torch
    6     from typing import Dict, Any
    7     from amaie_core.api.observation import Observation
    8 
    9     class PerceptionEngine:
   10         """
   11         Transforms raw time-series data into a rich, multi-scale feature set.
   12 
   13         This engine serves as the primary sensory processing unit for the
      AMAIE
   14         agent, packaging processed data into a standardized `Observation`
      object.
   15 
   16         Attributes:
   17             config: Configuration object for the perception engine.
   18         """
   19         def __init__(self, config: Any): # Use specific config type if 
      available
   20             self.config = config
   21 
   22         def process(self, raw_data: torch.Tensor, metadata: Dict[str, Any])
      -> Observation:
   23             """
   24             Processes a raw time-series tensor to generate a rich
      Observation.
   25 
   26             This method performs the core function of the engine:
   27             1. Validates and reshapes the input data.
   28             2. Computes relevant features (e.g., scattering spectra
      coefficients).
   29             3. Packages the features into a standardized Observation object.
   30 
   31             Args:
   32                 raw_data: A torch.Tensor containing the raw time-series data.
   33                           Expected shape should be compatible with the engine
      's config.
   34                 metadata: An optional dictionary of metadata to include in
      the
   35                           Observation object (e.g., data source, symbol).
   36 
   37             Returns:
   38                 An Observation object containing the computed features.
   39 
   40             Raises:
   41                 ValueError: If the input data shape is incompatible.
   42                 RuntimeError: If the feature extraction process fails.
   43             """
   44             # ... implementation ...
   45             return Observation(features={}, metadata={},
      timestamp=datetime.now(timezone.utc))

  ---

  5. Comments

  Use comments sparingly. They should explain why something is done, not what is done (which
  should be clear from the code itself).

   * Avoid: Redundant comments that re-state the obvious.
   * Use for:
       * Explaining complex algorithms or business logic.
       * Justifying non-obvious design choices or workarounds.
       * Highlighting potential pitfalls or areas for future improvement (TODO, FIXME).
   * Formatting: Use # for single-line comments. For multi-line comments, use multiple #
     lines or a docstring if it applies to a block of code.

   * Example:

   1     # This workaround is necessary due to a known bug in library X version Y.
   2     # TODO: Remove this once library X is updated to version Z.
   3     if some_condition:
   4         # Perform a complex calculation to minimize EFE, balancing epistemic 
     and pragmatic value.
   5         result = calculate_efe(state, action)
   6     else:
   7         # Fallback to a simpler heuristic for performance reasons in 
     low-latency scenarios.
   8         result = calculate_heuristic_value(state, action)

  ---

  6. Imports

   * Order: Imports should be grouped and ordered as follows:
       1. Standard library imports (e.g., os, sys, datetime).
       2. Third-party imports (e.g., torch, numpy, pytest).
       3. AMAIE project-specific imports (e.g., amaie_core.api.observation).
   * Alphabetical: Within each group, imports should be sorted alphabetically.
   * Absolute vs. Relative: Prefer absolute imports (e.g., from amaie_core.api.observation 
     import Observation) over relative imports (e.g., from .api.observation import 
     Observation) for clarity and to avoid ambiguity, especially in larger projects.
   * One Import Per Line: Generally, one import per line is preferred for readability.

   * Example:

    1     import asyncio
    2     import logging
    3     from datetime import datetime
    4     from typing import Any, Dict, List
    5 
    6     import click
    7     import torch
    8     from rich.console import Console
    9 
   10     from amaie_core.api.observation import Observation
   11     from amaie_core.components.perception import PerceptionEngine
   12     from amaie_core.config.perception import PerceptionConfig

  ---

  7. Error Handling

   * Specific Exceptions: Catch specific exceptions rather than broad Exception where
     possible.
   * Custom Exceptions: Define custom exceptions for domain-specific errors to provide
     clearer error messages and allow for more granular error handling.
   * Logging: Log exceptions with appropriate severity levels (error, critical) and include
     traceback information.
   * Graceful Degradation: Design components to fail gracefully and recover where possible.

   * Example:

    1     class ModelLoadingError(Exception):
    2         """Custom exception for errors during model loading."""
    3         pass
    4 
    5     try:
    6         model_definition = self.parser.load_model(model_file)
    7     except FileNotFoundError as e:
    8         logger.error(f"Model file not found: {model_file}", exc_info=True)
    9         raise ModelLoadingError(f"Model file '{model_file}' not found.") from
      e
   10     except yaml.YAMLError as e:
   11         logger.error(f"Invalid YAML in model file: {model_file}", exc_info=
      True)
   12         raise ModelLoadingError(f"Invalid YAML format in '{model_file}'.")
      from e
   13     except Exception as e:
   14         logger.critical(f"Unexpected error during model loading: {e}",
      exc_info=True)
   15         raise ModelLoadingError(f"An unexpected error occurred while loading 
      model: {e}") from e

  ---

  8. Code Formatting (Automated)

   * Black: The project uses black for automated code formatting. All code must be formatted
     with black.
   * Pre-commit Hooks: Consider using pre-commit hooks to automatically run black (and isort,
     flake8, mypy, pyright) before commits.

