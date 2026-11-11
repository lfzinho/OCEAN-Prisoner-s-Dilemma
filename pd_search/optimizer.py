"""Alternating pattern search over two OCEAN vectors + final frozen evaluation."""
from __future__ import annotations

import threading

import axelrod as axl
import numpy as np

from . import events as ev
from .config import Config
from .game import Emit, Stopped, play_match

NAMES = ["A", "B"]


class Optimizer:
    def __init__(self, cfg: Config, persona, emit: Emit, stop: threading.Event | None = None):
        self.cfg, self.persona, self.emit = cfg, persona, emit
        self.stop = stop or threading.Event()
        s = cfg.search
        self.rng = np.random.default_rng(cfg.model.base_seed)
        if s.init == "fixed":
            self.x = [np.array(s.agent_a_init, float), np.array(s.agent_b_init, float)]
        else:
            self.x = [self.rng.uniform(s.trait_min, s.trait_max, 5) for _ in range(2)]
        self.lr = [s.initial_lr, s.initial_lr]
        self.games = 0
        self.step = 0
        self.accepted = [0, 0]
        self.rejected = [0, 0]
        self.phase = "starting"

    # -- helpers ---------------------------------------------------------
    def _state(self):
        self.emit(ev.State([v.tolist() for v in self.x], list(self.lr), self.games, self.step,
                           list(self.accepted), list(self.rejected), self.phase))

    def _play(self, label: str, xa, xb, seed: int) -> tuple[float, float]:
        """Play one LLM-vs-LLM match. (Tests override this.)"""
        self.games += 1
        self.phase = label
        self._state()
        r = play_match(self.games, label, self.persona.describe(xa), self.persona.describe(xb),
                       self.cfg, seed, self.emit, self.stop)
        self._state()
        return r.scores

    def _clip(self, v):
        return np.clip(v, self.cfg.search.trait_min, self.cfg.search.trait_max)

    def active(self, p: int) -> bool:
        return self.lr[p] >= self.cfg.search.min_lr

    # -- search ----------------------------------------------------------
    def run(self) -> dict | None:
        try:
            self.search()
            return self.final_eval()
        except Stopped:
            return None

    def search(self) -> None:
        s, base = self.cfg.search, self.cfg.model.base_seed
        p = 0
        while self.active(0) or self.active(1):
            if not self.active(p):
                p = 1 - p
            self.step += 1
            seed = base + 1000 * self.step  # same seed for baseline + candidates (common random numbers)
            opp = 1 - p

            def play(label, cand):
                xs = [self.x[0], self.x[1]]
                xs[p] = cand
                return self._play(f"{NAMES[p]} step {self.step} {label}", xs[0], xs[1], seed)[p]

            f0 = play("baseline", self.x[p])
            best_f, best_x = f0, None
            dirs = self.rng.normal(size=(s.n_directions, 5))
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
            for i, d in enumerate(dirs, 1):
                cand = self._clip(self.x[p] + self.lr[p] * d)
                f = play(f"candidate {i}/{s.n_directions}", cand)
                if f > best_f:
                    best_f, best_x = f, cand
            if best_x is not None:
                self.x[p] = best_x
                self.accepted[p] += 1
            else:
                self.lr[p] /= 2
                self.rejected[p] += 1
            self.phase = f"step {self.step} done"
            self.emit(ev.StepEnd(self.step, p, best_x is not None, f0, best_f, self.x[p].tolist(), self.lr[p]))
            self._state()
            p = opp

    # -- final evaluation ------------------------------------------------
    def final_eval(self) -> dict:
        fe, base = self.cfg.final_eval, self.cfg.model.base_seed
        texts = [self.persona.describe(v) for v in self.x]
        sa, sb = [], []
        for k in range(fe.n_matches):
            r = self._final_match(f"final {k + 1}/{fe.n_matches}", texts[0], texts[1], base + 10**6 + k)
            sa.append(r[0]); sb.append(r[1])
        refs = {}
        for name in fe.reference_strategies:
            for slot, tag in enumerate(NAMES):
                scores = []
                for k in range(fe.reference_matches):
                    self.games += 1
                    self.phase = f"{tag} vs {name} {k + 1}/{fe.reference_matches}"
                    self._state()
                    r = play_match(self.games, self.phase, texts[slot], getattr(axl, name)(),
                                   self.cfg, base + 2 * 10**6 + k, self.emit, self.stop)
                    scores.append(r.scores[0])
                refs[f"{tag} vs {name}"] = {"mean": float(np.mean(scores)), "std": float(np.std(scores))}
        summary = {
            "agents": {
                tag: {"vector": self.x[i].tolist(), "persona": texts[i], "lr": self.lr[i],
                      "mean_score": float(np.mean(s)), "std_score": float(np.std(s)), "scores": s}
                for i, (tag, s) in enumerate(zip(NAMES, (sa, sb)))
            },
            "reference_scores": refs,
            "matches_played": self.games,
            "steps": self.step,
            "max_possible_per_match": self.cfg.game.rounds_per_match * self.cfg.game.payoff.T,
        }
        self.phase = "finished"
        self._state()
        self.emit(ev.Final(summary))
        return summary

    def _final_match(self, label, ta, tb, seed):
        self.games += 1
        self.phase = label
        self._state()
        return play_match(self.games, label, ta, tb, self.cfg, seed, self.emit, self.stop).scores
