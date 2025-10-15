"""
Unit tests for the GenerativeModelConfigLoader.
"""

import pytest
import yaml
import os

from sasaie_core.config.loader import GenerativeModelConfigLoader

@pytest.fixture
def valid_config_path(tmp_path):
    content = {
        'model': {
            'scales': [
                {
                    'name': 'fast',
                    'couplings': [
                        {'type': 's2_gated_top_down', 'from_scale': 'medium'}
                    ]
                },
                {
                    'name': 'medium',
                    'couplings': [
                        {'type': 's2_gated_bottom_up', 'from_scale': 'fast'},
                        {'type': 's2_gated_top_down', 'from_scale': 'slow'}
                    ]
                },
                {
                    'name': 'slow',
                    'couplings': [
                        {'type': 's2_gated_bottom_up', 'from_scale': 'medium'}
                    ]
                }
            ]
        }
    }
    path = tmp_path / "valid_config.yaml"
    with open(path, 'w') as f:
        yaml.dump(content, f)
    return str(path)

@pytest.fixture
def circular_config_path(tmp_path):
    content = {
        'model': {
            'scales': [
                {'name': 'fast', 'couplings': [{'type': 'simple_bottom_up', 'from_scale': 'slow'}]},
                {'name': 'medium', 'couplings': [{'type': 'simple_bottom_up', 'from_scale': 'fast'}]},
                {'name': 'slow', 'couplings': [{'type': 'simple_bottom_up', 'from_scale': 'medium'}]}
            ]
        }
    }
    path = tmp_path / "circular_config.yaml"
    with open(path, 'w') as f:
        yaml.dump(content, f)
    return str(path)

@pytest.fixture
def invalid_dep_config_path(tmp_path):
    content = {
        'model': {
            'scales': [
                {'name': 'fast', 'couplings': [{'type': 'simple_bottom_up', 'from_scale': 'non_existent'}]}
            ]
        }
    }
    path = tmp_path / "invalid_dep.yaml"
    with open(path, 'w') as f:
        yaml.dump(content, f)
    return str(path)

def test_successful_loading(valid_config_path):
    """Tests that a valid configuration is loaded correctly."""
    loader = GenerativeModelConfigLoader(valid_config_path)
    assert loader.config is not None
    assert len(loader.scales) == 3
    scale_config = loader.get_scale_config('medium')
    assert isinstance(scale_config['couplings'], list)
    assert len(scale_config['couplings']) == 2

def test_correct_execution_plan(valid_config_path):
    """Tests that the topological sort produces the correct execution order."""
    loader = GenerativeModelConfigLoader(valid_config_path)
    # Expected order: fast (no bottom-up deps), then medium (deps on fast), then slow (deps on medium)
    assert loader.execution_plan == ['fast', 'medium', 'slow']

def test_circular_dependency_error(circular_config_path):
    """Tests that a circular dependency raises a ValueError."""
    with pytest.raises(ValueError, match="Circular dependency detected"):
        GenerativeModelConfigLoader(circular_config_path)

def test_undefined_dependency_error(invalid_dep_config_path):
    """Tests that a dependency on a non-existent scale raises a ValueError."""
    with pytest.raises(ValueError, match="contains an undefined dependency"):
        GenerativeModelConfigLoader(invalid_dep_config_path)
