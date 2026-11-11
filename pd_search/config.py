from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ModelCfg(BaseModel):
    name: str
    temperature: float = 0.8
    base_seed: int = 42


class Payoff(BaseModel):
    T: float = 5
    R: float = 3
    P: float = 1
    S: float = 0


class GameCfg(BaseModel):
    rounds_per_match: int = Field(10, ge=1)
    talk_turns_per_round: int = Field(3, ge=0)
    move_fallback: Literal["COOPERATE", "DEFECT"] = "COOPERATE"
    payoff: Payoff = Payoff()
    rules_prompt: str
    talk_prompt: str
    move_prompt: str


class SearchCfg(BaseModel):
    trait_min: float = -1.0
    trait_max: float = 1.0
    initial_lr: float = 0.5
    min_lr: float = 0.03
    n_directions: int = Field(4, ge=1)
    init: Literal["random", "fixed"] = "random"
    agent_a_init: list[float] = [0.0] * 5
    agent_b_init: list[float] = [0.0] * 5

    @model_validator(mode="after")
    def _check(self):
        for v in (self.agent_a_init, self.agent_b_init):
            if len(v) != 5:
                raise ValueError("init vectors need 5 values (O, C, E, A, N)")
        if self.min_lr <= 0 or self.initial_lr <= 0:
            raise ValueError("learning rates must be > 0")
        return self


class FinalEvalCfg(BaseModel):
    n_matches: int = 5
    reference_strategies: list[str] = []
    reference_matches: int = 2


class PersonaRef(BaseModel):
    config: str = "persona.toml"


class Config(BaseModel):
    model: ModelCfg
    game: GameCfg
    search: SearchCfg
    final_eval: FinalEvalCfg = FinalEvalCfg()
    persona: PersonaRef = PersonaRef()
    base_dir: Path = Path(".")

    def persona_path(self) -> Path:
        return self.base_dir / self.persona.config


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return Config(**raw, base_dir=path.resolve().parent)
