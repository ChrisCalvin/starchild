SASAIE Model Structure Report

  This report defines the standardized structure and conventions for the YAML
  files that declaratively define the `HierarchicalRegimeVQVAE` world model
  within the SASAIE system. These model definitions are consumed by the
  ModelManager (`sasaie_core/components/model.py`) to construct the agent's
  probabilistic understanding of its environment.

  Adhering to this structure ensures consistency and allows for the flexible definition of hierarchical world models.

  ---

  1. Core Principles

   * Declarative Definition: World models are defined using human-readable YAML
     files, separating model architecture from implementation logic.
   * Hierarchical & Compositional: The structure must naturally represent the multi-level, compositional nature of the `HierarchicalRegimeVQVAE`.
   * Modularity: Each level of the hierarchy should be a self-contained, configurable unit.
   * Readability: YAML files should be well-commented and logically organized.

  ---

  2. Overall YAML Structure

  A model definition YAML file must contain a top-level `model` key.

   ```yaml
   model:
     model_name: "string"        # Unique name for the model
     description: "string"       # Human-readable description
     input_source: "string"      # ID of the feature vector from the Perception Engine (e.g., "matrix_profile")
     levels:                   # List defining the hierarchy, from bottom-up
       - # LevelDefinition
   ```

  ---

  3. `levels` Section

  The `levels` section is the core of the model definition. It is a list where each item defines a single level of the VQ-VAE hierarchy. The list should be ordered from the bottom layer (closest to the input) to the top layer (most abstract).

  Each item in the `levels` list must conform to the `LevelDefinition` schema:

   ```yaml
   - name: "string"              # Unique name for the level (e.g., "fine_grained_patterns")
     description: "string"     # Description of what this level is intended to learn
     codebook:
       size: integer           # Number of discrete codes (regimes) in this level's library
       dim: integer            # Dimensionality of each codebook vector
     transition_model: "string"  # Type of temporal transition model (e.g., "LSTM", "Transformer")
     composition:                # Optional: Defines how this level composes from the one below
       from_level: "string"    # Name of the level below to compose from
       type: "string"          # Composition type (e.g., "cross_attention")
       config: {}              # Specific configuration for the composition type (e.g., attention heads)
   ```

  **Attributes Explained:**

   * `name` (Required, string): A unique string identifier for the level.
   * `description` (Required, string): A clear explanation of the level's role.
   * `codebook` (Required, object): Defines the properties of this level's Vector-Quantized codebook.
       * `size` (Required, integer): The number of discrete latent vectors in the codebook. This is the size of this level's "library of regimes."
       * `dim` (Required, integer): The dimensionality of each latent vector in the codebook.
   * `transition_model` (Required, string): Specifies the neural network architecture used to model temporal dynamics *within* this level (i.e., predicting the next regime at this same scale).
   * `composition` (Optional, object): If present, defines how this level abstracts patterns from the level below it. This is omitted for the first level, which connects directly to the input.
       * `from_level` (Required, string): The `name` of the level this level is attending to.
       * `type` (Required, string): The mechanism for composition. `cross_attention` is the standard.
       * `config` (Optional, object): A key-value map for parameters of the composition mechanism, e.g., `num_heads: 8` for cross-attention.

  ---

  4. Example Model YAML (`generative_model.yaml`)

  This example demonstrates a three-level `HierarchicalRegimeVQVAE`.

   ```yaml
   model:
     model_name: "MarketRegimeVQVAE_v1"
     description: "A 3-level hierarchical VQ-VAE for detecting market regimes from a Matrix Profile."
     input_source: "matrix_profile_summary"

     levels:
       # Level 0: Bottom layer, connects to the input from the Perception Engine
       - name: "fine_grained_motifs"
         description: "Learns a codebook of fine-grained, high-frequency patterns from the Matrix Profile."
         codebook:
           size: 512
           dim: 64
         transition_model: "LSTM"

       # Level 1: Middle layer, composes from the fine_grained_motifs layer
       - name: "medium_term_trends"
         description: "Learns a codebook of trends by attending to sequences of fine-grained motifs."
         codebook:
           size: 256
           dim: 128
         transition_model: "LSTM"
         composition:
           from_level: "fine_grained_motifs"
           type: "cross_attention"
           config:
             num_heads: 8

       # Level 2: Top layer, most abstract
       - name: "macro_regimes"
         description: "Learns a codebook of long-term market states (e.g., bull, bear, sideways)."
         codebook:
           size: 64
           dim: 256
         transition_model: "LSTM"
         composition:
           from_level: "medium_term_trends"
           type: "cross_attention"
           config:
             num_heads: 4
   ```

  ---

  5. Best Practices

   * Bottom-Up Definition: Define levels in the YAML file from the lowest (most granular) to the highest (most abstract).
   * Descriptive Names: Use clear and concise names for the model and its levels.
   * Align Dimensions: While not explicitly linked, ensure the `dim` of a lower level's codebook is compatible with the expectations of the level composing it.
   * Version Control: Keep model YAMLs under version control to track architectural experiments and changes.

