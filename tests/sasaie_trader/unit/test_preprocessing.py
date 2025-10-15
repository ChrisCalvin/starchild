# Part of the new RegimeVAE architecture as of 2025-10-13

import pytest
import numpy as np
import stumpy

from sasaie_trader.preprocessing import (
    HierarchicalStreamingMP,
    StreamingRegimeDetector,
)

# Constants for testing
TEST_SCALES = [10, 20]
INITIAL_TS_LENGTH = 100

@pytest.fixture
def initial_ts() -> np.ndarray:
    """Provides a sample initial time series."""
    return np.random.randn(INITIAL_TS_LENGTH)


# --- Test Cases for HierarchicalStreamingMP --- #

class TestHierarchicalStreamingMP:
    @pytest.fixture
    def hmp(self, initial_ts) -> HierarchicalStreamingMP:
        """Provides a HierarchicalStreamingMP instance."""
        return HierarchicalStreamingMP(initial_ts=initial_ts, scales=TEST_SCALES)

    def test_hmp_instantiation(self, hmp):
        """Test that the streaming matrix profile objects are created."""
        assert hmp is not None
        assert set(hmp.streams.keys()) == set(TEST_SCALES)
        assert set(hmp.current_mps.keys()) == set(TEST_SCALES)
        # Check that the initial matrix profiles have the correct length
        for scale in TEST_SCALES:
            assert len(hmp.current_mps[scale]) == INITIAL_TS_LENGTH - scale + 1

    def test_hmp_update(self, hmp):
        """Test the update method returns a correctly structured dictionary."""
        # Arrange
        new_point = np.random.randn()

        # Act
        updated_mps = hmp.update(new_point)

        # Assert
        assert isinstance(updated_mps, dict)
        assert set(updated_mps.keys()) == set(TEST_SCALES)
        for scale in TEST_SCALES:
            assert isinstance(updated_mps[scale], np.ndarray)
            # The length of the matrix profile should grow with the data
            assert len(updated_mps[scale]) == (INITIAL_TS_LENGTH + 1) - scale

    def test_hmp_get_latest_profile(self, hmp):
        """Test retrieving the latest profile for a specific scale."""
        # Arrange
        scale_to_check = TEST_SCALES[0]
        hmp.update(np.random.randn())
    
        # Act
        profile = hmp.get_latest_profile(scale_to_check)
    
        # Assert
        assert isinstance(profile, np.ndarray)
        assert len(profile) == (INITIAL_TS_LENGTH + 1) - scale_to_check

# --- Test Cases for StreamingRegimeDetector --- #

class TestStreamingRegimeDetector:
    @pytest.fixture
    def srd(self, initial_ts) -> StreamingRegimeDetector:
        """Provides a StreamingRegimeDetector instance."""
        return StreamingRegimeDetector(initial_ts=initial_ts, m=TEST_SCALES[0])

    def test_srd_instantiation(self, srd):
        """Test that the stumpi stream object is created."""
        assert srd is not None
        assert hasattr(srd, 'stream')
        assert hasattr(srd.stream, 'update')

    def test_srd_update(self, srd):
        """Test that the update method returns the correct data types."""
        # Arrange
        new_point = np.random.randn()

        # Act
        result = srd.update(new_point)

        # Assert
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], (bool, np.bool_))
        assert isinstance(result[1], float)