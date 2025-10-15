# Part of the new RegimeVAE architecture as of 2025-10-13

# Complete Data Flow

## Single Timestep Pipeline

This document outlines the step-by-step data flow for a single observation entering the system.

```
New Observation
    ↓
[1] Update Streaming Matrix Profile (O(n))
    - The latest data point is pushed to the STUMPY streaming object.
    - An updated matrix profile is computed efficiently.
    ↓
[2] Encode to Regime Codes (hierarchically)
    - The matrix profile is preprocessed and fed into the HierarchicalRegimeVQVAE.
    - The model encodes the pattern at each level, producing a discrete code and a distance metric.
    ├─ Level 1: code=3, distance=0.25
    ├─ Level 2: code=7, distance=0.31  
    └─ Level 3: code=2, distance=0.18
    ↓
[3] Check for Novelty (distance > threshold?)
    - The distance from the VQ-VAE encoding is used as a novelty score.
    ├─ NO → Use existing code. The belief update proceeds normally.
    └─ YES → A novel pattern has been discovered. Trigger an adaptation.
        - Add a new code to the VQ-VAE codebook.
        - Fine-tune the VQ-VAE on the new pattern using EWC.
    ↓
[4] Update Expert Models
    - The new observation is passed to the expert forecaster corresponding to the current regime code.
    └─ expert_bank[code].update(observation)
    ↓
[5] Generate Forecasts
    - The relevant expert model generates a forecast for the near future.
    └─ forecast = expert_bank[code].predict(horizon=10)
    ↓
[6] Update Beliefs
    - The planner updates its structured beliefs about the world state.
    ├─ Regime confidence (derived from VQ-VAE distance)
    ├─ Transition imminence (from composition matrices)
    └─ Expected duration (from historical analysis)
    ↓
[7] Generate Candidate Policies
    - Based on the current beliefs, the planner generates a set of potential actions.
    ├─ Exploit (if confident)
    ├─ Explore (if uncertain)
    ├─ Prepare (if transition likely)
    └─ Adapt (if novel)
    ↓
[8] Compute EFE for each policy
    - Each candidate policy is evaluated by calculating its Expected Free Energy.
    - EFE = (Expected Reward) - (Information Gain)
    └─ Select the policy with the minimum EFE.
    ↓
[9] Execute Action
    - The first action from the optimal policy is taken.
    ↓
[10] Learn from Outcome
    - The result of the action is observed and used to update the system.
    └─ Update regime-policy associations (e.g., if an action worked well in a regime, reinforce it).
```

**Total Time**: ~10-50ms per timestep (depending on series length and planning complexity).