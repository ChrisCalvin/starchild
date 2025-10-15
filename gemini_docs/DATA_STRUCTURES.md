AMAIE Core Data Structure Report

  This report outlines the primary data structures used within the amaie-core
  package, which represent the fundamental language of the Active Inference
  agent. These structures facilitate the flow of information between the
  agent's perception, modeling, and planning components.

  ---

  1. Observation

  Description: Represents a standardized, domain-agnostic input to the world
  model. This object is the output of the PerceptionEngine and serves as the
  primary input to the ModelManager. It contains processed features from raw
  sensory data.

  Attributes:

   * features: Dict[str, torch.Tensor] - A dictionary of feature names to PyTorch
     tensors. This is the core data, containing processed information like
     scattering spectra coefficients.
   * metadata: Dict[str, Any] - A dictionary for any ancillary information, such
     as the data source, symbol, or other contextual details that are not part of
     the model's feature space.
   * timestamp: datetime - The UTC timestamp when the observation was generated.

  Template:

    1 from datetime import datetime
    2 import torch
    3 from amaie_core.api.observation import Observation
    4 
    5 # Example: A simple observation with a single feature 
      'market_state_t1'
    6 observation_template = Observation(
    7     features={
    8         "market_state_t1": torch.tensor([0.5, -0.2],
      dtype=torch.float32)
    9     },
   10     metadata={
   11         "source": "live_market_data",
   12         "symbol": "BTC/USDT",
   13         "interval": "1m"
   14     },
   15     timestamp=datetime.now(timezone.utc)
   16 )

  ---

  2. BeliefState

  Description: Represents the agent's current understanding of the world state
  after processing observations through the generative model. It contains
  probability distributions over hidden variables, uncertainty estimates, and
  confidence measures. This object is the output of world model inference.

  Attributes:

   * distributions: Dict[str, torch.distributions.Distribution] - A dictionary
     mapping hidden variable names to their inferred probability distributions
     (from torch.distributions).
   * uncertainty: torch.Tensor - A tensor quantifying the epistemic uncertainty
     (model uncertainty) associated with the beliefs.
   * confidence: float - An overall confidence score (0.0 to 1.0) in the accuracy
     of this belief state.
   * metadata: Dict[str, Any] - A dictionary for any ancillary information from
     the inference process, such as convergence status or iteration count.
   * timestamp: datetime - The UTC timestamp when the belief state was computed.
   * belief_id: str - A unique identifier for this specific belief state instance.
   * observation_id: Optional[str] - The ID of the Observation that led to this
     belief state.

  Template:


    1 from datetime import datetime, timezone
    2 import torch
    3 from torch.distributions import Normal, MultivariateNormal
    4 from amaie_core.api.belief import BeliefState
    5 
    6 # Example: A belief state with a normal distribution for 
      'market_state_t0'
    7 belief_state_template = BeliefState(
    8     distributions={
    9         "market_state_t0": MultivariateNormal(
   10             loc=torch.tensor([0.1, 0.05]),
   11             covariance_matrix=torch.eye(2) * 0.01
   12         )
   13     },
   14     uncertainty=torch.tensor(0.05),
   15     confidence=0.95,
   16     metadata={"inference_steps": 100},
   17     timestamp=datetime.now(timezone.utc),
   18     observation_id="some_observation_uuid"
   19 )

  ---

  3. Action

  Description: Represents a concrete, executable action that can be performed
  by an external system (e.g., trading engine, robotics controller). It's
  derived from a Policy but contains specific execution details and
  constraints.

  Attributes:

   * command: str - The specific command identifier for the external system (e.g.,
     "create_order", "cancel_order").
   * params: Dict[str, Any] - A dictionary of execution parameters (e.g., symbol,
     side, amount).
   * priority: ActionPriority (Enum) - The execution priority level (LOW, NORMAL,
     HIGH, CRITICAL).
   * timeout: Optional[float] - An optional maximum execution time in seconds.
   * status: ActionStatus (Enum) - The current execution status of the action
     (PENDING, EXECUTING, COMPLETED, FAILED, CANCELLED).
   * metadata: Dict[str, Any] - A dictionary for any additional execution-related
     metadata.
   * timestamp: datetime - The UTC timestamp when the action was created.
   * action_id: str - A unique identifier for this action instance.
   * policy_id: Optional[str] - The ID of the Policy that generated this action.
   * execution_result: Optional[Dict[str, Any]] - The result of the action's
     execution.

  Template:

    1 from datetime import datetime
    2 from amaie_core.api.action import Action, ActionPriority,
      ActionStatus
    3 
    4 # Example: A trading action to create a buy order
    5 action_template = Action(
    6     command="create_order",
    7     params={
    8         "symbol": "BTC/USDT",
    9         "side": "buy",
   10         "amount": 0.01,
   11         "order_type": "limit",
   12         "price": 30000.0
   13     },
   14     priority=ActionPriority.HIGH,
   15     timeout=10.0,
   16     status=ActionStatus.PENDING,
   17     metadata={"source_module": "amaie_trader.execution"},
   18     timestamp=datetime.now(timezone.utc),
   19     policy_id="some_policy_uuid"
   20 )

  ---

  4. Policy

  Description: Represents a high-level action plan selected by the Active
  Inference planner. It contains the action type, parameters, expected
  outcomes, and confidence measures. Policies are domain-agnostic descriptions
  of what the agent intends to do.

  Attributes:

   * actions: list[Action] - A list of Action objects representing the sequence
     of actions to be taken.
   * expected_outcome: Optional[BeliefState] - The predicted BeliefState if this
     policy is executed.
   * confidence: float - The agent's confidence (0.0 to 1.0) in this policy
     selection.
   * expected_free_energy: float - The calculated EFE value for this policy.
   * planning_horizon: int - The number of future time steps this policy covers.
   * metadata: Dict[str, Any] - A dictionary for ancillary data, such as the
     breakdown of EFE into epistemic and pragmatic values.
   * timestamp: datetime - The UTC timestamp when the policy was selected.
   * policy_id: str - A unique identifier for this policy instance.
   * belief_state_id: Optional[str] - The ID of the BeliefState that led to this
     policy.

  Template:

    1 from datetime import datetime
    2 import torch
    3 from torch.distributions import Normal, MultivariateNormal
    4 from amaie_core.api.action import Action, ActionPriority
    5 from amaie_core.api.belief import BeliefState
    6 from amaie_core.api.policy import Policy
    7 
    8 # Example: A policy with a single action and an expected outcome
    9 policy_template = Policy(
   10     actions=[
   11         Action(
   12             command="hold_position",
   13             params={"reason": "awaiting_confirmation"},
   14             priority=ActionPriority.LOW
   15         )
   16     ],
   17     expected_outcome=BeliefState(
   18         distributions={
   19             "market_state_t0": MultivariateNormal(
   20                 loc=torch.tensor([0.1, 0.05]),
   21                 covariance_matrix=torch.eye(2) * 0.01
   22             )
   23         },
   24         uncertainty=torch.tensor(0.04),
   25         confidence=0.98
   26     ),
   27     confidence=0.85,
   28     expected_free_energy=-0.123,
   29     planning_horizon=1,
   30     metadata={"pragmatic_value": -0.1, "epistemic_value": -0.023},
   31     timestamp=datetime.now(timezone.utc),
   32     belief_state_id="some_belief_state_uuid"
   33 )

Ground Truth for External Data Integration (e.g., scatspectra)

  The PerceptionEngine (in amaie_core/components/perception.py) is responsible
  for taking raw data and transforming it into an Observation object. When
  using libraries like scatspectra, the output of these libraries needs to be
  mapped to the features dictionary of the Observation.

   1. Source Library Output (`scatspectra`):
       * Variable Name (internal to `scatspectra`): The
         scatspectra.frontend.analyze function returns a DescribedTensor object.
         The core numerical data (the scattering coefficients) is typically
         accessed via its .y attribute.
       * Format: described_tensor.y is a numpy.ndarray. Its shape depends on the
         input data and the scattering transform parameters (J, Q, etc.). For a
         single time series, it might be (num_coefficients,) or (batch_size, 
         num_coefficients).

   2. Transformation and Integration into `Observation`:
       * The PerceptionEngine takes the numpy.ndarray from described_tensor.y and
         converts it into a torch.Tensor.
       * Crucially, this torch.Tensor is then assigned to a specific key within
         the Observation.features dictionary. This key must match a variable id
         defined in the generative model's YAML configuration (e.g.,
         configs/generative_model.yaml).

   3. Expected "Ground Truth" Name and Format within `Observation.features`:
       * Variable Name (as seen by `ModelManager`): Based on
         configs/generative_model.yaml, the ModelManager expects a feature named
         "market_state_t1".
       * Format: torch.Tensor.
       * Expected Shape/Dimension: The generative_model.yaml defines
         market_state_t1 with a dimension: 2. Therefore, the torch.Tensor
         associated with "market_state_t1" in the Observation.features dictionary
         should have a shape compatible with this dimension (e.g., (batch_size, 2)
          or simply (2,) if processing a single instance).

