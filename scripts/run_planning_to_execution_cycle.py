import logging
import torch
from torch.distributions import Normal
import sys
import os

# Add project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sasaie_core.components.planner import Planner
from sasaie_trader.execution import ExecutionEngine
from sasaie_core.api.belief import BeliefState

def setup_logging():
    """Sets up basic logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )

def run_cycle():
    """Runs a single planning-to-execution cycle."""
    logger = logging.getLogger(__name__)
    logger.info("--- Starting Planning-to-Execution Cycle ---")

    # 1. Instantiate components
    # Define a sample configuration for the new planner
    planner_config = {
        "model_config_path": "configs/generative_model.yaml",
        "num_candidate_policies": 5,
        "planning_horizon": 3,
        "vmp_iterations": 10,
        "policy_selector_config": {
            "input_dim": 2,
            "latent_dim": 8,
            "hidden_dim": 32,
            "vocab_size": 4,  # Let's say we have 4 different morphisms
            "context_dim": 4,
        },
        "goals": {
            "maximize_return": {
                "loc": [1.0, 0.0],  # Target a state with high positive trend
                "scale": [0.1, 0.1],
                "prefixes": ["state_"],  # This goal applies to all state nodes
                "weight": 1.0
            }
        }
    }
    planner = Planner(config=planner_config)
    execution_engine = ExecutionEngine()

    # 2. Create a mock BeliefState
    # In a real scenario, this would come from the Global Belief Fusion component.
    # Using a Normal distribution to be compatible with the current VMP Message format.
    mock_distributions = {
        "market_state_t0": Normal(
            loc=torch.tensor([0.1, 0.05]),
            scale=torch.tensor([0.1, 0.1])  # scale = sqrt(variance)
        )
    }
    belief_state = BeliefState(
        distributions=mock_distributions,
        uncertainty=torch.tensor(0.05),
        confidence=0.95
    )
    logger.info(f"Created mock belief state: {belief_state.belief_id}")

    # 3. Call the planner to get an optimal policy
    policy = planner.plan(belief_state)

    # 4. Extract the first action from the policy
    if not policy.actions:
        logger.warning("Planner returned a policy with no actions. Ending cycle.")
        return

    action_to_execute = policy.actions[0]
    logger.info(f"Extracted action {action_to_execute.action_id} from policy {policy.policy_id}")

    # 5. Call the execution engine with the action
    result_action = execution_engine.execute_action(action_to_execute)

    # 6. Print the final result
    logger.info(f"--- Cycle Complete ---")
    logger.info(f"Final action status: {result_action.status.value}")
    logger.info(f"Execution result: {result_action.execution_result}")

if __name__ == "__main__":
    setup_logging()
    run_cycle()
