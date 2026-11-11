"""Events streamed from the optimizer/matches to the TUI and the JSONL log."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class MatchStart:
    match_id: int
    label: str
    persona_a: str
    persona_b: str
    total_units: int  # messages + moves expected in this match


@dataclass
class Message:
    match_id: int
    round: int
    slot: int  # 0 = agent A, 1 = agent B
    text: str


@dataclass
class Move:
    match_id: int
    round: int
    slot: int
    move: str  # "C" / "D"
    raw: str = ""


@dataclass
class Round:
    match_id: int
    round: int
    moves: tuple[str, str]
    payoffs: tuple[float, float]


@dataclass
class MatchEnd:
    match_id: int
    scores: tuple[float, float]
    coop: tuple[float, float]


@dataclass
class StepEnd:
    step: int
    player: int
    accepted: bool
    baseline: float
    best: float
    vector: list[float]
    lr: float


@dataclass
class State:
    x: list[list[float]]
    lr: list[float]
    games: int
    step: int
    accepted: list[int]
    rejected: list[int]
    phase: str


@dataclass
class Final:
    summary: dict = field(default_factory=dict)


def to_dict(ev) -> dict:
    return {"type": type(ev).__name__, **asdict(ev)}
