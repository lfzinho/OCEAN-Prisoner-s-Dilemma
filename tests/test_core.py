import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from persona_writer import PersonaWriter  # noqa: E402
from pd_search import events as ev  # noqa: E402
from pd_search.config import load_config  # noqa: E402
from pd_search.game import parse_move  # noqa: E402
from pd_search.optimizer import Optimizer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def test_parse_move():
    assert str(parse_move("I will COOPERATE")) == "C"
    assert str(parse_move("defect.")) == "D"
    assert parse_move("hmm") is None


def test_persona_grounding_interpolates():
    w = PersonaWriter(ROOT / "persona.toml")
    g = w.grounding([0.3, -1.0, 0.0, 1.0, 0.25])
    assert "60% of the way from position +0.0 to position +0.5" in g  # openness 0.3
    assert "Strongly prefers" not in g and "Very spontaneous" in g     # exact anchor at -1.0
    assert "50% of the way from position +0.0 to position +0.5" in g   # neuroticism 0.25


class FakeOpt(Optimizer):
    """Deterministic quadratic game: each agent wants to be close to its own target."""
    targets = [np.array([1, 0, 0, 0, 0.0]), np.array([0, 0, 0, 0, -1.0])]

    def _play(self, label, xa, xb, seed):
        self.games += 1
        return (-float(np.sum((xa - self.targets[0]) ** 2)), -float(np.sum((xb - self.targets[1]) ** 2)))


def test_optimizer_converges_and_halves_lr():
    cfg = load_config(ROOT / "config.toml")
    cfg.search.n_directions = 6
    evs = []
    o = FakeOpt(cfg, None, evs.append)
    o.search()
    assert o.lr[0] < cfg.search.min_lr and o.lr[1] < cfg.search.min_lr
    for i in range(2):
        assert np.linalg.norm(o.x[i] - FakeOpt.targets[i]) < 0.5
        assert o.accepted[i] > 0 and o.rejected[i] > 0
    steps = [e for e in evs if isinstance(e, ev.StepEnd)]
    assert [s.player for s in steps[:4]] == [0, 1, 0, 1]  # alternates
    for s in steps:
        if not s.accepted:
            assert s.best == s.baseline
    assert all(-1 <= v <= 1 for x in o.x for v in x)
