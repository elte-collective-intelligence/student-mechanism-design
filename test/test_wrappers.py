"""Integration tests for compliant agent wrappers."""

from src.agent import GATAgent, TransformerAgent, GNNAgent, BaseAgent


def test_gat_wrapper_compliance():
    """Verify GATAgent satisfies BaseAgent and works correctly."""
    agent = GATAgent(node_feature_size=5, hidden_dim=16)

    assert isinstance(agent, BaseAgent)
    assert agent.agent_type == "gat"

    # Test method existence (delegation check)
    assert hasattr(agent, "select_action")
    assert hasattr(agent, "update")
    assert hasattr(agent, "save")
    assert hasattr(agent, "load")


def test_transformer_wrapper_compliance():
    """Verify TransformerAgent satisfies BaseAgent and works correctly."""
    agent = TransformerAgent(node_feature_size=5, hidden_dim=16)

    assert isinstance(agent, BaseAgent)
    assert agent.agent_type == "transformer"

    # Test method existence
    assert hasattr(agent, "select_action")
    assert hasattr(agent, "update")


def test_gnn_wrapper_compliance():
    """Verify GNNAgent satisfies BaseAgent and works correctly."""
    agent = GNNAgent(node_feature_size=5, hidden_dim=16)

    assert isinstance(agent, BaseAgent)
    assert agent.agent_type == "gnn"

    # Test method existence
    assert hasattr(agent, "select_action")
    assert hasattr(agent, "update")
    assert hasattr(agent, "save")
    assert hasattr(agent, "load")
