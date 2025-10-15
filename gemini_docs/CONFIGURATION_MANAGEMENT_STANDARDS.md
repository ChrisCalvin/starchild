AMAIE Configuration Management Standards

  This document defines the standards for managing application configurations within the
  AMAIE system. Effective configuration management is crucial for deploying, operating,
  and maintaining the system across various environments (development, testing, paper
  trading, live trading) while ensuring security and flexibility.

  ---

  1. Core Principles

   * Separation of Configuration from Code: Configuration values should be externalized from
     the codebase, allowing changes without modifying and redeploying code.
   * Environment-Specific Configuration: Support different configurations for different
     environments (e.g., API keys for live vs. paper trading, logging levels).
   * Security for Sensitive Data: Sensitive information (secrets) must be handled securely
     and never committed to version control.
   * Hierarchical Loading: Configurations should be loaded in a defined order, allowing for
     overrides based on environment or specific profiles.
   * Validation: Configurations should be validated at load time to catch errors early.
   * Readability: Configuration files should be human-readable and well-structured.

  ---

  2. Configuration Sources and Hierarchy

  AMAIE uses a layered approach to configuration loading, where later sources can
  override earlier ones.

   1. Default Values (Codebase): Hardcoded defaults within the application logic. These are
      the lowest priority.
   2. Base Configuration Files (`configs/`): General, non-sensitive configuration values
      stored in YAML files.
       * configs/core/: Core AMAIE agent configuration.
       * configs/trader/: Trading-specific configuration.
       * configs/launcher/: Launcher-specific configuration.
       * configs/generative_model.yaml: Declarative world model definition.
   3. Profile-Specific Configuration (`configs/launcher/launch_profiles/`): YAML files
      defining specific deployment profiles (e.g., development.yaml, paper_trading.yaml,
      live_trading.yaml). These override base configurations.
   4. Environment Variables: Used for sensitive data (secrets) and environment-specific
      overrides. These have the highest priority.

  ---

  3. Configuration File Formats

   * YAML: The primary format for structured configuration files (.yaml or .yml). YAML is
     preferred for its human-readability and support for complex data structures.
   * `.env` files: Used for local development to manage environment variables without
     setting them globally. These files should never be committed to version control.

  ---

  4. Managing Sensitive Data (Secrets)

  Sensitive information (e.g., API keys, exchange credentials, database passwords) must be
   handled with extreme care.

   * Never Commit Secrets: Secrets must never be hardcoded in the codebase or committed to
     version control (Git).
   * Environment Variables: The primary mechanism for injecting secrets into the
     application at runtime.
       * For local development, use a .env file (e.g., THOR_EXCHANGE_API_KEY=your_key).
       * For production, use the deployment environment's secret management capabilities
         (e.g., Docker secrets, Kubernetes secrets, cloud provider secret managers).
   * Access Control: Ensure that only authorized personnel and services have access to
     sensitive configuration.

   * Example (`.env` file for local development):

    1     # .env (DO NOT COMMIT TO GIT)
    2     # This file is for local development environment variables.
    3 
    4     # Exchange API Credentials
    5     HUMMINGBOT_API_KEY=your_hummingbot_api_key_dev
    6     HUMMINGBOT_SECRET_KEY=your_hummingbot_secret_key_dev
    7     BINANCE_API_KEY=your_binance_api_key_dev
    8     BINANCE_SECRET_KEY=your_binance_secret_key_dev
    9 
   10     # Database Credentials
   11     DB_USER=amaie_dev
   12     DB_PASSWORD=dev_password123
   13 
   14     # Other sensitive settings
   15     EMAIL_SENDER_PASSWORD=your_email_password

  ---

  5. Profile-Specific Configuration

  The launcher uses profiles to define different operational modes. Each profile is a
  YAML file that specifies settings relevant to that mode.

   * Location: configs/launcher/launch_profiles/<profile_name>.yaml
   * Content:
       * name: Unique name of the profile.
       * description: Human-readable description.
       * trading_mode: (e.g., development, paper, live).
       * exchange: (e.g., binance, kucoin).
       * trading_pair: (e.g., BTC-USDT).
       * References to other configuration files or specific overrides.
       * Mode-specific settings (e.g., development might enable debug features, live might
         enforce stricter safety checks).

   * Example (`configs/launcher/launch_profiles/paper_trading.yaml`):

    1     name: "paper_trading"
    2     description: "Paper trading profile for simulated market conditions."
    3     trading_mode: "paper"
    4     trading_pair: "ETH-USDT"
    5     exchange: "binance_paper_testnet" # Specific mock exchange or testnet
    6 
    7     # Overrides for core agent configuration
    8     amaie_core_config:
    9       model:
   10         device: "cpu" # Force CPU for paper trading
   11       planner:
   12         cem:
   13           num_iterations: 5 # Faster planning for simulation
   14 
   15     # Overrides for trader configuration
   16     amaie_trader_config:
   17       execution:
   18         simulate_latency: true
   19         simulate_slippage: true
   20       risk:
   21         max_drawdown: 10.0 # Higher drawdown tolerance for paper trading
   22 
   23     # Launcher specific settings
   24     log_level: "INFO"
   25     enable_dashboard: true

  ---

  6. Configuration Loading and Access

   * Centralized Loading: A dedicated ConfigurationManager (in launcher/config_manager.py)
     is responsible for loading and merging configurations from various sources.
   * Type-Safe Access: Configuration values should be accessed through type-safe objects
     (e.g., Pydantic models) to ensure correct types and enable validation.
   * Lazy Loading: Load configurations only when needed to minimize startup time and
     resource consumption.

   * Example (Pydantic for config schema):

    1     from pydantic import BaseModel, Field
    2     from typing import Optional
    3 
    4     class ModelConfig(BaseModel):
    5         device: str = "cuda"
    6         dtype: str = "float32"
    7         model_file: str = "generative_model.yaml"
    8 
    9     class PlannerCEMConfig(BaseModel):
   10         num_iterations: int = 10
   11         elite_fraction: float = 0.2
   12 
   13     class PlannerConfig(BaseModel):
   14         planning_horizon: int = 5
   15         cem: PlannerCEMConfig = Field(default_factory=PlannerCEMConfig)
   16 
   17     class CoreConfig(BaseModel):
   18         model: ModelConfig = Field(default_factory=ModelConfig)
   19         planner: PlannerConfig = Field(default_factory=PlannerConfig)
   20 
   21     # In ConfigurationManager:
   22     # config_data = load_and_merge_yaml_files(...)
   23     # core_config = CoreConfig(**config_data.get("amaie_core_config", {}))

  ---

  7. Configuration Validation

   * Schema Validation: Use libraries like Pydantic to define schemas for configuration
     structures and automatically validate loaded data.
   * Semantic Validation: Implement custom validation logic for inter-dependencies or
     business rules (e.g., max_drawdown must be positive).
   * Early Failure: Fail fast if configuration is invalid, providing clear error messages.

  ---

