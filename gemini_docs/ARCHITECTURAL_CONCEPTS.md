# Part of the new RegimeVAE architecture as of 2025-10-13

# Core Architectural Concepts

---

## 1. Core Components

### 1.1 Streaming Matrix Profile (Perception)

**Purpose**: Detect patterns and regime changes in O(n) time

**Key Features**:
- Updates incrementally as new data arrives (not O(n²) recomputation)
- FLOSS algorithm for online regime change detection
- Multi-scale analysis (10, 50, 200 step windows)
- Discord detection for novelty

### 1.2 Hierarchical VQ-VAE (World Model)

**Purpose**: Learn discrete regime library with compositional structure

**Architecture**:
```
Level 3 (Long-term Regimes, m=200)
    ↑ conditions on ↑
Level 2 (Medium Patterns, m=50)
    ↑ conditions on ↑
Level 1 (Short Rules, m=10)
```

**Key Features**:
- Separate codebook per scale
- Cross-attention between levels
- Composition matrices learn emergence rules
- Dynamic codebook growth for novel regimes

### 1.3 Continual Learning (Evolution)

**Purpose**: Learn new regimes without forgetting old ones

**Method**: Elastic Weight Consolidation (EWC)
- Computes Fisher Information Matrix on old regimes
- Penalizes changes to "important" parameters
- Fine-tunes on novel patterns in ~5 epochs

### 1.4 Expert Bank (Forecasting)

**Purpose**: Regime-specific forecasters

**Structure**:
```python
expert_bank = {
    regime_0: ARForecaster(),  # Trending expert
    regime_1: ARForecaster(),  # Volatile expert
    ...
}
```

### 1.5 Active Inference Planner (Action Selection)

**Purpose**: Minimize Expected Free Energy across timescales

**EFE Calculation**:
`EFE = Pragmatic Value - Epistemic Value`

**Policy Types**:
1. **EXPLOIT**: High confidence → use learned strategy
2. **EXPLORE**: Low confidence → gather information
3. **PREPARE**: Transition imminent → conservative actions
4. **ADAPT**: Novel regime → cautious strategy

---

## 2. Hierarchical Composition Learning

### 2.1 The "Ruliad" Insight

**Question**: Do short-window motifs (rules) generate long-term patterns?

**Answer**: YES. The system learns how simple rules compose into complex patterns.

### 2.2 Composition Matrix

Learns `P(pattern_k | rules_{k-1})`:

```python
composition_matrix[i, j] = probability that
    rule j at level k-1
    participates in pattern i at level k
```

---

## 3. Advantages Over Alternatives

### 3.1 vs. Single VQ-VAE (Flat)

| Aspect | Flat VQ-VAE | Hierarchical |
|--------|-------------|--------------|
| Interpretability | Opaque codes | Clear hierarchy |
| Long-term modeling | Poor | Excellent |
| Emergence learning | No | Yes |

### 3.2 vs. Pure Matrix Profile

| Aspect | Matrix Profile | MP + VQ-VAE |
|--------|----------------|-------------|
| Forecasting | ❌ None | ✅ Generative |
| Compact representation | ❌ High-dim | ✅ Single code |
| Continuous learning | ❌ No | ✅ Yes |

### 3.3 vs. Deep Learning (LSTM/Transformer)

| Aspect | Deep Learning | This System |
|--------|---------------|-------------|
| Interpretability | Black box | Regime labels |
| Data efficiency | Needs lots | Works with little |
| Online adaptation | Slow/unstable | Fast + stable (EWC) |