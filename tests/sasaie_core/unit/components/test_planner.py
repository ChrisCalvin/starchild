# Part of the new RegimeVAE architecture as of 2025-10-13

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock

from sasaie_core.components.planner import RegimeAwarePlanner
from sasaie_core.api.belief import RegimeBeliefs
from sasaie_core.api.action import ActionType
from sasaie_core.api.policy import Policy

# Constants for testing
TEST_SCALES = [10, 50]
TEST_CODEBOOK_SIZES = [8, 4]

# --- Mocks and Stubs --- #

@pytest.fixture
def mock_vqvae():
    """Provides a mock of the HierarchicalRegimeVQVAE model."""
    vqvae = MagicMock()
    # Mock the layer structure to access codebook sizes
    vqvae.layers = [MagicMock(), MagicMock()]
    vqvae.layers[0].codebook_size = TEST_CODEBOOK_SIZES[0]
    vqvae.layers[1].codebook_size = TEST_CODEBOOK_SIZES[1]
    return vqvae

@pytest.fixture
def mock_expert_bank():
    """Provides a mock expert bank."""
    return {i: MagicMock() for i in range(max(TEST_CODEBOOK_SIZES))}

@pytest.fixture
def planner(mock_vqvae, mock_expert_bank) -> RegimeAwarePlanner:
    """Provides a RegimeAwarePlanner instance with mocked dependencies."""
    return RegimeAwarePlanner(
        scales=TEST_SCALES,
        vqvae=mock_vqvae,
        expert_bank=mock_expert_bank
    )

# --- Test Cases --- #

class TestRegimeAwarePlanner:

    def test_planner_instantiation(self, planner):
        """Test that the planner and its components are instantiated correctly."""
        assert planner is not None
        assert planner.scales == TEST_SCALES
        assert hasattr(planner, 'efe_calculator')

    def test_update_beliefs(self, planner):
        """Test the conversion of raw regime codes into structured Belief objects."""
        # Arrange
        regime_codes = {
            10: (3, 0.1),  # scale -> (code, distance)
            50: (1, 0.4)
        }

        # Act
        beliefs = planner.update_beliefs(regime_codes)

        # Assert
        assert isinstance(beliefs, dict)
        assert set(beliefs.keys()) == {10, 50}
        
        belief_10 = beliefs[10]
        assert isinstance(belief_10, RegimeBeliefs)
        assert belief_10.current_regime == 3
        assert pytest.approx(belief_10.regime_confidence) == 1.0 / 1.1
        assert belief_10.regime_probabilities.shape == (TEST_CODEBOOK_SIZES[0],)

    @pytest.mark.parametrize("confidence, expected_policy_type", [
        (0.8, ActionType.EXPLOIT), # High confidence -> EXPLOIT
        (0.5, ActionType.EXPLORE), # Low confidence -> EXPLORE
    ])
    def test_generate_candidate_policies_confidence(self, planner, confidence, expected_policy_type):
        """Test that policy generation is sensitive to regime confidence."""
        # Arrange
        beliefs = {
            10: RegimeBeliefs(current_regime=1, regime_confidence=confidence, regime_probabilities=torch.ones(8), expected_duration=10.0, transition_imminence=0.1)
        }

        # Act
        policies = planner.generate_candidate_policies(beliefs)

        # Assert
        assert any(p.action_type == expected_policy_type for p in policies)

    def test_generate_candidate_policies_transition_risk(self, planner):
        """Test that a PREPARE policy is generated when transition risk is high."""
        # Arrange
        beliefs = {
            10: RegimeBeliefs(current_regime=1, regime_confidence=0.9, regime_probabilities=torch.ones(8), expected_duration=10.0, transition_imminence=0.7)
        }

        # Act
        policies = planner.generate_candidate_policies(beliefs)

        # Assert
        # Should generate both EXPLOIT (due to high confidence) and PREPARE (due to high risk)
        action_types = {p.action_type for p in policies}
        assert ActionType.EXPLOIT in action_types
        assert ActionType.PREPARE in action_types

    def test_select_best_policy(self, planner, mocker):
        """Test that the policy with the minimum EFE is selected."""
        # Arrange
        # Mock the EFE calculator to return predictable values
        mocker.patch.object(planner.efe_calculator, 'compute_efe', side_effect=[10.0, 5.0, 12.0])

        dummy_beliefs = {10: MagicMock()}
        dummy_forecasts = {10: np.zeros(10)}

        # Mock policy generation to return a fixed set of candidates
        candidates = [
            Policy(action_type=ActionType.EXPLOIT, actions=[], horizon=10),
            Policy(action_type=ActionType.EXPLORE, actions=[], horizon=10),
            Policy(action_type=ActionType.PREPARE, actions=[], horizon=10),
        ]
        mocker.patch.object(planner, 'generate_candidate_policies', return_value=candidates)

        # Act
        best_policy = planner.select_best_policy(dummy_beliefs, dummy_forecasts)

        # Assert
        # The second policy, which got an EFE of 5.0, should be chosen
        assert best_policy.action_type == ActionType.EXPLORE
        assert best_policy.efe == 5.0

    def test_learn_from_outcome(self, planner):
        """Test that a successful policy is cached for the correct regime."""
        # Arrange
        regime = 3
        regime_codes = {10: (regime, 0.1)}
        good_policy = Policy(action_type=ActionType.EXPLOIT, actions=[0.1, 0.2], horizon=2)
        # An outcome close to 0.5 is considered good in the planner's reward function
        good_outcome = 0.51 

        # Act
        planner.learn_from_outcome(good_policy, good_outcome, regime_codes)

        # Assert
        assert regime in planner.regime_policy_cache
        assert planner.regime_policy_cache[regime] == good_policy.actions
