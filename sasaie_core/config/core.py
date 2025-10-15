from pydantic import BaseModel, Field
from typing import Optional

class ModelConfig(BaseModel):
    device: str = "cuda"
    dtype: str = "float32"
    model_file: str = "generative_model.yaml"

class PlannerCEMConfig(BaseModel):
    num_iterations: int = 10
    elite_fraction: float = 0.2

class PlannerConfig(BaseModel):
    planning_horizon: int = 5
    cem: PlannerCEMConfig = Field(default_factory=PlannerCEMConfig)

class CoreConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    planner: PlannerConfig = Field(default_factory=PlannerConfig)
