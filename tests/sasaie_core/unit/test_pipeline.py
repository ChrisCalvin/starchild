# Part of the new RegimeVAE architecture as of 2025-10-13

import pytest
import numpy as np
import json
from unittest.mock import MagicMock, patch
import logging
import torch

from sasaie_core.pipeline import MainPipeline
from sasaie_core.components.planner import RegimeAwarePlanner
from sasaie_core.components.perception import HierarchicalStreamingMP # Changed import path
from sasaie_core.api.belief import RegimeBeliefs
from sasaie_core.api.policy import Policy
from sasaie_core.api.action import ActionType

# Constants for testing
TEST_SCALES = [10, 50]
TEST_MP_INPUT_DIM = 100
TEST_INITIAL_BUFFER_SIZE = 500

@pytest.fixture
def mock_vqvae():
    vqvae = MagicMock()
    vqvae.hierarchical_encode.return_value = {10: (1, 0.1), 50: (2, 0.2)}
    return vqvae

@pytest.fixture
def mock_planner():
    planner = MagicMock(spec=RegimeAwarePlanner)
    # Mock update_beliefs
    mock_beliefs = {
        10: RegimeBeliefs(current_regime=1, regime_confidence=0.9, regime_probabilities=torch.tensor([0.1, 0.9]), expected_duration=10.0, transition_imminence=0.1),
        50: RegimeBeliefs(current_regime=2, regime_confidence=0.8, regime_probabilities=torch.tensor([0.2, 0.8]), expected_duration=20.0, transition_imminence=0.2)
    }
    planner.update_beliefs.return_value = mock_beliefs
    
    # Mock generate_forecasts
    mock_forecasts = {
        10: np.array([0.5, 0.6, 0.7]),
        50: np.array([1.0, 1.1, 1.2])
    }
    planner.generate_forecasts.return_value = mock_forecasts

    # Mock select_best_policy
    planner.select_best_policy.return_value = Policy(action_type=ActionType.EXPLOIT, actions=[0.1], horizon=1)
    
    return planner

@pytest.fixture
def main_pipeline(mock_vqvae, mock_planner):
    return MainPipeline(
        vqvae_model=mock_vqvae,
        planner=mock_planner,
        scales=TEST_SCALES,
        mp_input_dim=TEST_MP_INPUT_DIM,
        initial_buffer_size=TEST_INITIAL_BUFFER_SIZE
    )

class TestMainPipeline:

    def test_init(self, main_pipeline):
        assert main_pipeline.vqvae is not None
        assert main_pipeline.planner is not None
        assert main_pipeline.scales == TEST_SCALES
        assert main_pipeline.mp_input_dim == TEST_MP_INPUT_DIM
        assert main_pipeline.initial_buffer_size == TEST_INITIAL_BUFFER_SIZE
        assert len(main_pipeline.ts_buffer) == 0
        assert main_pipeline.mp_stream is None

    def test_preprocess_mp(self, main_pipeline):
        mp_data = np.array([i for i in range(200)])
        processed = main_pipeline._preprocess_mp(mp_data)
        assert processed.shape == (1, TEST_MP_INPUT_DIM)
        assert isinstance(processed, torch.Tensor)

        mp_data_short = np.array([i for i in range(50)])
        processed_short = main_pipeline._preprocess_mp(mp_data_short)
        assert processed_short.shape == (1, TEST_MP_INPUT_DIM)
        assert isinstance(processed_short, torch.Tensor)

    @patch('sasaie_core.pipeline.HierarchicalStreamingMP')
    def test_process_warmup(self, MockHierarchicalStreamingMP, main_pipeline):
        # Simulate warm-up period
        for i in range(TEST_INITIAL_BUFFER_SIZE - 1):
            main_pipeline.process(json.dumps({'price': 100.0 + i}).encode())
            assert main_pipeline.mp_stream is None
        
        # Last item to trigger initialization
        main_pipeline.process(json.dumps({'price': 100.0 + TEST_INITIAL_BUFFER_SIZE - 1}).encode())
        assert main_pipeline.mp_stream is not None
        MockHierarchicalStreamingMP.assert_called_once()

    @patch('sasaie_core.pipeline.HierarchicalStreamingMP')
    def test_process_full_cycle(self, MockHierarchicalStreamingMP, main_pipeline, mock_vqvae, mock_planner):
        # Initialize MP stream
        main_pipeline.mp_stream = MockHierarchicalStreamingMP.return_value
        main_pipeline.mp_stream.update.return_value = {10: np.zeros(10), 50: np.zeros(50)}

        # Simulate a data point
        raw_payload = json.dumps({'price': 105.0}).encode()
        main_pipeline.process(raw_payload)

        # Assertions for each step of the pipeline
        main_pipeline.mp_stream.update.assert_called_once_with(105.0)
        mock_vqvae.hierarchical_encode.assert_called_once()
        mock_planner.update_beliefs.assert_called_once_with(mock_vqvae.hierarchical_encode.return_value)
        mock_planner.generate_forecasts.assert_called_once_with(mock_vqvae.hierarchical_encode.return_value, horizon=10)
        mock_planner.select_best_policy.assert_called_once_with(mock_planner.update_beliefs.return_value, mock_planner.generate_forecasts.return_value)
        mock_planner.learn_from_outcome.assert_called_once_with(mock_planner.select_best_policy.return_value, 105.0, mock_vqvae.hierarchical_encode.return_value)

    def test_process_invalid_payload(self, main_pipeline, caplog):
        with caplog.at_level(logging.WARNING):
            main_pipeline.process(b'invalid json')
            assert "Failed to decode JSON from MQTT message." in caplog.text
        
        with caplog.at_level(logging.WARNING):
            main_pipeline.process(json.dumps({'no_price': 123}).encode())
            assert "Received MQTT message without a 'price' field." in caplog.text
