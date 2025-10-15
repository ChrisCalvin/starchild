"""
Unit tests for the Forney Factor Graph data structures and VMP updates.
"""

import pytest
import torch
from unittest.mock import MagicMock

from sasaie_core.planning.graph import Node, Factor, FactorGraph, Message, GaussianFactor

@pytest.fixture
def empty_graph():
    """Provides an empty FactorGraph instance."""
    return FactorGraph()

@pytest.fixture
def graph_with_nodes(empty_graph):
    """Provides a FactorGraph with some nodes added."""
    graph = empty_graph
    graph.add_node("state_t0", dim=2)
    graph.add_node("action_t0", dim=1)
    graph.add_node("state_t1", dim=2)
    return graph

def test_node_creation():
    """Tests basic Node initialization."""
    node = Node("test_node", dim=2)
    assert node.name == "test_node"
    assert node.dim == 2
    assert node.belief is None
    assert node.factors == []

def test_add_node(empty_graph):
    """Tests adding a node to the graph."""
    node = empty_graph.add_node("new_node", dim=2)
    assert "new_node" in empty_graph.nodes
    assert empty_graph.get_node("new_node") == node

def test_add_duplicate_node_raises_error(empty_graph):
    """Tests that adding a node with a duplicate name raises a ValueError."""
    empty_graph.add_node("duplicate_node", dim=1)
    with pytest.raises(ValueError):
        empty_graph.add_node("duplicate_node", dim=1)

def test_add_factor(graph_with_nodes):
    """Tests adding a factor to the graph."""
    factor = graph_with_nodes.add_factor("transition_factor", ["state_t0", "action_t0", "state_t1"])
    assert "transition_factor" in graph_with_nodes.factors
    state_t0 = graph_with_nodes.get_node("state_t0")
    assert factor in state_t0.factors

def test_node_update_rule():
    """Tests the VMP update rule for a Node."""
    # 1. Setup
    node = Node("test_node", dim=1)
    factor1 = MagicMock(spec=Factor)
    factor1.name = "f1"
    factor2 = MagicMock(spec=Factor)
    factor2.name = "f2"
    
    node.factors = [factor1, factor2]
    
    # Incoming messages from two factors
    msg1 = Message(loc=torch.tensor([1.0]), scale=torch.tensor([1.0]))
    msg2 = Message(loc=torch.tensor([3.0]), scale=torch.tensor([2.0]))
    node.incoming_messages = {"f1": msg1, "f2": msg2}

    # 2. Expected result (analytical product of two Gaussians)
    # Precision τ = 1/σ^2
    tau1 = 1.0 / (1.0 ** 2)
    tau2 = 1.0 / (2.0 ** 2)
    expected_total_tau = tau1 + tau2  # 1.0 + 0.25 = 1.25
    
    # μ_new * τ_new = μ1*τ1 + μ2*τ2
    expected_loc = (1.0 * tau1 + 3.0 * tau2) / expected_total_tau # (1.0 + 0.75) / 1.25 = 1.4
    expected_scale = torch.sqrt(torch.tensor(1.0 / expected_total_tau)) # sqrt(1 / 1.25) = sqrt(0.8) = 0.8944

    # 3. Act
    node.update()

    # 4. Assert
    assert node.belief is not None
    assert torch.isclose(node.belief.loc, torch.tensor([expected_loc]))
    assert torch.isclose(node.belief.scale, expected_scale)

def test_gaussian_factor_update():
    """Tests the VMP update rule for a GaussianFactor."""
    # 1. Setup
    x_node = Node(name="x", dim=1)
    y_node = Node(name="y", dim=1)
    b = torch.tensor([5.0])
    noise_scale = torch.tensor([3.0])
    factor = GaussianFactor(name="f_xy", nodes=[x_node, y_node], b=b, noise_scale=noise_scale)

    # 2. Test message sending from x to y
    # Mock a message coming into the factor from node x
    msg_from_x = Message(loc=torch.tensor([10.0]), scale=torch.tensor([4.0]))
    factor.receive_message(from_node_name="x", message=msg_from_x)
    
    # Act
    factor.update()

    # Assert: Check the message received by y
    msg_to_y = y_node.incoming_messages.get("f_xy")
    assert msg_to_y is not None
    
    # Expected loc = E[x] + b = 10.0 + 5.0 = 15.0
    expected_loc_y = torch.tensor([15.0])
    # Expected scale = sqrt(Var(x) + Var(noise)) = sqrt(4^2 + 3^2) = sqrt(16 + 9) = sqrt(25) = 5.0
    expected_scale_y = torch.tensor([5.0])

    assert torch.isclose(msg_to_y.loc, expected_loc_y)
    assert torch.isclose(msg_to_y.scale, expected_scale_y)

    # 3. Test message sending from y to x
    # Clear previous incoming messages for a clean test
    factor.incoming_messages = {}
    
    # Mock a message coming into the factor from node y
    msg_from_y = Message(loc=torch.tensor([20.0]), scale=torch.tensor([6.0]))
    factor.receive_message(from_node_name="y", message=msg_from_y)

    # Act
    factor.update()

    # Assert: Check the message received by x
    msg_to_x = x_node.incoming_messages.get("f_xy")
    assert msg_to_x is not None

    # Expected loc = E[y] - b = 20.0 - 5.0 = 15.0
    expected_loc_x = torch.tensor([15.0])
    # Expected scale = sqrt(Var(y) + Var(noise)) = sqrt(6^2 + 3^2) = sqrt(36 + 9) = sqrt(45) = 6.708
    expected_scale_x = torch.tensor([6.7082])

    assert torch.isclose(msg_to_x.loc, expected_loc_x)
    assert torch.isclose(msg_to_x.scale, expected_scale_x, atol=1e-4)