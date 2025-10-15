# Gemini Project: Starlord

This file provides context for Gemini to understand and assist with this project.

## Executive Summary

This system combines **Streaming Matrix Profile**, **Hierarchical VQ-VAE**, and **Active Inference** to create an adaptive, interpretable framework for time series forecasting and control. It learns the "grammar of emergence" - how short-term rules compose into long-term regimes.

### Key Innovation
Instead of a single complex model, we maintain a **dynamically growing library of expert models**, where each expert specializes in a specific regime. The system automatically discovers new regimes and learns their hierarchical structure.

---

## Integration with sasaie_core

### Component Mapping

| sasaie_core Component | Current | New | Benefit |
|----------------------|---------|-----|---------|
| **World Model** | MultiScaleNFModel | HierarchicalRegimeVQVAE | Discrete regimes, interpretable |
| **Perception** | Custom | StreamingMP + FLOSS | O(n) updates, auto-detect changes |
| **Planner** | EFE minimization | RegimeAwarePlanner | Hierarchical, anticipates transitions |
| **Evolution** | Structural learning | Dynamic codebook + EWC | Grows without forgetting |

### Configuration

Replace `configs/generative_model.yaml` with:

```yaml
model_type: hierarchical_regime_vqvae

scales:
  - name: short_term
    window_size: 10
    codebook_size: 16
    
  - name: medium_term
    window_size: 50
    codebook_size: 12
    conditions_on: short_term
    
  - name: long_term
    window_size: 200
    codebook_size: 8
    conditions_on: medium_term

perception:
  type: streaming_matrix_profile
  use_floss: true
  novelty_threshold: 0.3

planner:
  type: regime_aware_active_inference
  efe_calculation: hierarchical

evolution:
  enabled: true
  codebook_growth: dynamic
  continual_learning: ewc
```

---

## Building and Running

*   **Build, Run, Test** For detailed development cycle, see `gemini_docs/BUILD_RUN_TEST_INSTRUCTIONS.md`
*   **Project Roadmap** For a high-level development plan and milestones, see `gemini_docs/PROJECT_ROADMAP.md`

## Development Conventions

*   **Data Structures:** For detailed data structure definitions and templates, refer to `g>
*   **API Structures:** For comprehensive API design guidelines and examples, see `gemini_d>
*   **Coding Standards:** For Python coding style and conventions, refer to `gemini_docs/CO>
*   **Testing Strategy:** For testing types, conventions, and execution, including the use >
*   **Logging & Monitoring:** For logging formats and monitoring guidelines, refer to `gemi>
*   **Configuration Management:** For how configurations are managed, see `gemini_docs/CONF>
*   **Dependency Management:** For dependency declaration and management, refer to `gemini_>

