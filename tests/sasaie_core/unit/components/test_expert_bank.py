# Part of the new RegimeVAE architecture as of 2025-10-13

import pytest
import numpy as np
from unittest.mock import MagicMock
from collections import deque
from typing import Dict, Any, Optional # Added imports

from sasaie_core.components.experts.base_expert import BaseExpert, ExpertMetadata
from sasaie_core.components.experts.ar_forecaster import ARForecaster
from sasaie_core.components.experts.expert_bank import (
    ExpertBankManager,
    ContinualExpertBank,
    create_ar_expert_factory,
    create_expert_bank
)

# Mock Expert for BaseExpert Protocol testing
class MockExpert:
    def __init__(self, regime_id: int, buffer_size: int = 100):
        self._regime_id = regime_id
        self._n_observations = 0
        self.history = deque(maxlen=buffer_size)
        self.params = {'regime_id': regime_id, 'value': 0.0}
        self.uncertainty = 0.1

    @property
    def regime_id(self) -> int:
        return self._regime_id

    @property
    def n_observations(self) -> int:
        return self._n_observations

    def update(self, observation: float, context: Optional[Dict[str, Any]] = None) -> None:
        self.history.append(observation)
        self._n_observations += 1
        self.params['value'] = observation # Simulate parameter update

    def predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self.params['value'])

    def get_parameters(self) -> Dict[str, Any]:
        return self.params

    def set_parameters(self, params: Dict[str, Any]) -> None:
        self.params = params
        self._regime_id = params['regime_id']

    def clone(self) -> 'MockExpert':
        new_expert = MockExpert(self._regime_id, self.history.maxlen)
        new_expert.set_parameters(self.get_parameters())
        return new_expert

    def get_uncertainty(self) -> float:
        return self.uncertainty


# ============================================================================
# Test BaseExpert Protocol
# ============================================================================

def test_base_expert_protocol_adherence():
    expert = MockExpert(regime_id=1)
    assert isinstance(expert, BaseExpert)
    assert expert.regime_id == 1
    assert expert.n_observations == 0
    expert.update(10.0)
    assert expert.n_observations == 1
    assert expert.predict(5).shape == (5,)
    assert isinstance(expert.get_parameters(), dict)
    expert.set_parameters({'regime_id': 2, 'value': 20.0})
    assert expert.regime_id == 2
    assert isinstance(expert.clone(), MockExpert)
    assert isinstance(expert.get_uncertainty(), float)


# ============================================================================
# Test ARForecaster
# ============================================================================

def test_ar_forecaster_init():
    ar_expert = ARForecaster(regime_id=1, order=5, min_observations=10, buffer_size=50)
    assert ar_expert.regime_id == 1
    assert ar_expert.order == 5
    assert ar_expert.min_observations == 10
    assert ar_expert.history.maxlen == 50
    assert ar_expert.coefficients is None

def test_ar_forecaster_update_and_fit():
    ar_expert = ARForecaster(regime_id=1, order=2, min_observations=5, buffer_size=10)
    
    # Not enough data to fit
    for i in range(4):
        ar_expert.update(float(i))
    assert ar_expert.coefficients is None
    
    # Enough data to fit
    for i in range(4, 10):
        ar_expert.update(float(i))
    
    # Should have fitted by now (due to periodic refit or explicit call)
    ar_expert.fit() # Ensure fit is called
    assert ar_expert.coefficients is not None
    assert len(ar_expert.coefficients) == ar_expert.order
    assert ar_expert.n_observations == 10

def test_ar_forecaster_predict():
    ar_expert = ARForecaster(regime_id=1, order=2, min_observations=5, buffer_size=10)
    data = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
    for d in data:
        ar_expert.update(d)
    
    forecast = ar_expert.predict(horizon=3)
    assert forecast.shape == (3,)
    # Basic check: forecast should be increasing if data is increasing
    assert forecast[0] > data[-1]
    assert forecast[1] > forecast[0]

def test_ar_forecaster_get_set_parameters():
    ar_expert = ARForecaster(regime_id=1, order=2, min_observations=5, buffer_size=10)
    data = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
    for d in data:
        ar_expert.update(d)
    ar_expert.fit()

    params = ar_expert.get_parameters()
    assert params['regime_id'] == 1
    assert params['order'] == 2
    assert isinstance(params['coefficients'], list)
    assert isinstance(params['intercept'], float)
    assert isinstance(params['residual_variance'], float)
    assert isinstance(params['history'], list)
    assert params['n_observations'] == 10

    new_ar_expert = ARForecaster(regime_id=99, order=1, min_observations=1, buffer_size=1)
    new_ar_expert.set_parameters(params)
    assert new_ar_expert.regime_id == 1
    assert new_ar_expert.order == 2
    assert new_ar_expert.n_observations == 10
    assert new_ar_expert.coefficients is not None

def test_ar_forecaster_clone():
    ar_expert = ARForecaster(regime_id=1, order=2, min_observations=5, buffer_size=10)
    data = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
    for d in data:
        ar_expert.update(d)
    ar_expert.fit()

    cloned_expert = ar_expert.clone()
    assert cloned_expert.regime_id == ar_expert.regime_id
    assert cloned_expert.order == ar_expert.order
    assert np.array_equal(cloned_expert.coefficients, ar_expert.coefficients)
    assert cloned_expert.intercept == ar_expert.intercept
    assert cloned_expert.residual_variance == ar_expert.residual_variance
    assert list(cloned_expert.history) == list(ar_expert.history)
    assert cloned_expert.n_observations == ar_expert.n_observations
    
    # Ensure it's a deep copy
    cloned_expert.update(2.0)
    assert cloned_expert.n_observations != ar_expert.n_observations

def test_ar_forecaster_uncertainty():
    ar_expert = ARForecaster(regime_id=1, order=2, min_observations=5, buffer_size=10)
    data = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
    for d in data:
        ar_expert.update(d)
    ar_expert.fit()
    assert ar_expert.get_uncertainty() >= 0.0


# ============================================================================
# Test ExpertBankManager
# ============================================================================

@pytest.fixture
def mock_expert_factory():
    def _factory(regime_id: int):
        return MockExpert(regime_id)
    return _factory

@pytest.fixture
def expert_bank_manager(mock_expert_factory):
    return ExpertBankManager(expert_factory=mock_expert_factory, max_experts=3, pruning_threshold=10)

def test_expert_bank_manager_init(expert_bank_manager):
    assert len(expert_bank_manager.experts) == 0
    assert expert_bank_manager.current_timestep == 0

def test_expert_bank_manager_get_or_create_expert(expert_bank_manager):
    expert1 = expert_bank_manager.get_or_create_expert(1)
    assert expert1.regime_id == 1
    assert len(expert_bank_manager.experts) == 1
    
    expert2 = expert_bank_manager.get_or_create_expert(1) # Get existing
    assert expert2 is expert1

    expert3 = expert_bank_manager.get_or_create_expert(2)
    assert expert3.regime_id == 2
    assert len(expert_bank_manager.experts) == 2

def test_expert_bank_manager_update_expert(expert_bank_manager):
    expert_bank_manager.update_expert(1, 10.0)
    expert = expert_bank_manager.experts[1]
    assert expert.n_observations == 1
    assert expert.get_parameters()['value'] == 10.0
    assert expert_bank_manager.metadata[1].n_updates == 1
    assert expert_bank_manager.metadata[1].last_active == 0 # Timestep not advanced yet

def test_expert_bank_manager_get_forecast(expert_bank_manager):
    expert_bank_manager.update_expert(1, 10.0)
    forecast = expert_bank_manager.get_forecast(1, 5)
    assert np.array_equal(forecast, np.full(5, 10.0))
    assert expert_bank_manager.total_forecasts == 1
    assert expert_bank_manager.metadata[1].last_active == 0

def test_expert_bank_manager_ensemble_forecast(expert_bank_manager):
    expert_bank_manager.update_expert(1, 10.0)
    expert_bank_manager.update_expert(2, 20.0)
    
    regime_probs = {1: 0.7, 2: 0.3}
    ensemble = expert_bank_manager.get_ensemble_forecast(regime_probs, 2)
    assert np.array_equal(ensemble, np.full(2, 0.7 * 10.0 + 0.3 * 20.0))

def test_expert_bank_manager_pruning(expert_bank_manager):
    expert_bank_manager.get_or_create_expert(1)
    expert_bank_manager.get_or_create_expert(2)
    expert_bank_manager.get_or_create_expert(3)
    assert len(expert_bank_manager.experts) == 3

    # Advance timestep to make expert 1 inactive
    expert_bank_manager.current_timestep = 15
    expert_bank_manager.update_expert(2, 5.0) # Keep expert 2 active
    expert_bank_manager.update_expert(3, 5.0) # Keep expert 3 active

    pruned_count = expert_bank_manager.prune_inactive_experts()
    assert pruned_count == 1 # Expert 1 should be pruned
    assert len(expert_bank_manager.experts) == 2
    assert 1 not in expert_bank_manager.experts

def test_expert_bank_manager_prune_least_used(expert_bank_manager):
    expert_bank_manager.get_or_create_expert(1)
    expert_bank_manager.get_or_create_expert(2)
    expert_bank_manager.get_or_create_expert(3)
    expert_bank_manager.get_or_create_expert(4) # This should trigger pruning
    assert len(expert_bank_manager.experts) == 3 # Max experts is 3
    assert 1 not in expert_bank_manager.experts # Expert 1 should be pruned as it has 0 updates

def test_expert_bank_manager_step(expert_bank_manager):
    initial_timestep = expert_bank_manager.current_timestep
    expert_bank_manager.step()
    assert expert_bank_manager.current_timestep == initial_timestep + 1

def test_expert_bank_manager_save_load_state(expert_bank_manager, mock_expert_factory):
    expert_bank_manager.update_expert(1, 10.0)
    expert_bank_manager.update_expert(2, 20.0)
    expert_bank_manager.step()
    
    state = expert_bank_manager.save_state()
    
    new_bank = ExpertBankManager(expert_factory=mock_expert_factory, max_experts=3, pruning_threshold=10)
    new_bank.load_state(state)
    
    assert new_bank.current_timestep == expert_bank_manager.current_timestep
    assert new_bank.total_forecasts == expert_bank_manager.total_forecasts
    assert len(new_bank.experts) == len(expert_bank_manager.experts)
    assert new_bank.experts[1].get_parameters() == expert_bank_manager.experts[1].get_parameters()


# ============================================================================
# Test ContinualExpertBank
# ============================================================================

@pytest.fixture
def continual_expert_bank(mock_expert_factory):
    return ContinualExpertBank(expert_factory=mock_expert_factory, max_experts=3, pruning_threshold=10, ewc_lambda=0.5)

def test_continual_expert_bank_init(continual_expert_bank):
    assert continual_expert_bank.ewc_lambda == 0.5

def test_continual_expert_bank_freeze_expert(continual_expert_bank):
    continual_expert_bank.get_or_create_expert(1)
    continual_expert_bank.freeze_expert(1)
    assert continual_expert_bank.metadata[1].is_frozen == True

def test_continual_expert_bank_compute_fisher_information(continual_expert_bank):
    # For MockExpert, fisher_information will be None as it's not an ARForecaster
    continual_expert_bank.get_or_create_expert(1)
    continual_expert_bank.compute_fisher_information(1)
    assert continual_expert_bank.metadata[1].fisher_information is None

    # Test with ARForecaster
    ar_factory = create_ar_expert_factory(order=2)
    ar_bank = ContinualExpertBank(expert_factory=ar_factory)
    ar_bank.get_or_create_expert(1)
    for i in range(20):
        ar_bank.update_expert(1, float(i))
    ar_bank.compute_fisher_information(1)
    assert ar_bank.metadata[1].fisher_information is not None
    assert 'coefficients' in ar_bank.metadata[1].fisher_information
    assert 'intercept' in ar_bank.metadata[1].fisher_information


# ============================================================================
# Test Factory Functions
# ============================================================================

def test_create_ar_expert_factory():
    factory = create_ar_expert_factory(order=5)
    expert = factory(regime_id=10)
    assert isinstance(expert, ARForecaster)
    assert expert.regime_id == 10
    assert expert.order == 5
    assert expert.min_observations == 20 # max(20, 5*2) = 20

def test_create_expert_bank_ar():
    bank = create_expert_bank(expert_type="ar", order=7, max_experts=5, pruning_threshold=50, ewc_lambda=0.1)
    assert isinstance(bank, ContinualExpertBank)
    assert bank.max_experts == 5
    assert bank.pruning_threshold == 50
    assert bank.ewc_lambda == 0.1
    expert = bank.expert_factory(1)
    assert isinstance(expert, ARForecaster)
    assert expert.order == 7

def test_create_expert_bank_unknown_type():
    with pytest.raises(ValueError, match="Unknown expert type: unknown"):
        create_expert_bank(expert_type="unknown")

def test_create_expert_bank_ffg_not_implemented():
    with pytest.raises(NotImplementedError, match="FFG experts not yet implemented"):
        create_expert_bank(expert_type="ffg")
