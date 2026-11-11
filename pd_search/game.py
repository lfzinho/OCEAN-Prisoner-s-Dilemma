"""One iterated PD match between LLM agents, running on top of Axelrod.

Axelrod handles the rounds, payoffs and match statistics. An LLMPlayer is an
axl.Player whose strategy() lets the (shared) ChatRoom run the pre-move chat
for the round and then asks the LLM for a private move.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Callable

import axelrod as axl
import ollama

from . import events as ev
from .config import Config

C, D = axl.Action.C, axl.Action.D
WORD = {C: "COOPERATE", D: "DEFECT"}
Emit = Callable[[object], None]


class Stopped(Exception):
    pass


def parse_move(text: str) -> axl.Action | None:
    m = re.search(r"COOPERATE|DEFECT", text.upper())
    if not m:
        return None
    return C if m.group(0) == "COOPERATE" else D


class ChatRoom:
    def __init__(self, cfg: Config, match_id: int, personas, seed: int, emit: Emit, stop: threading.Event | None):
        self.cfg, self.match_id, self.personas = cfg, match_id, personas  # personas[slot] is None for non-LLM
        self.seed, self.emit, self.stop = seed, emit, stop
        self.chat_enabled = all(p is not None for p in personas)
        self.game = axl.Game(**{k.lower(): v for k, v in cfg.game.payoff.model_dump().items()})
        self.players: list = [None, None]
        self.chat: list[tuple[int, str]] = []
        self.chatted_round = -1
        self.pending: dict[int, dict[int, axl.Action]] = {}
        self.n_calls = 0

    # -- LLM plumbing ----------------------------------------------------
    def _ask(self, slot: int, template: str, rnd: int) -> str:
        if self.stop is not None and self.stop.is_set():
            raise Stopped
        g = self.cfg.game
        pay = g.payoff
        system = g.rules_prompt.format(
            persona=self.personas[slot], rounds=g.rounds_per_match, talk_turns=g.talk_turns_per_round,
            R=pay.R, S=pay.S, T=pay.T, P=pay.P,
        )
        chat = "\n".join(f"{'You' if s == slot else 'Other person'}: {t}" for s, t in self.chat) or "(no messages yet)"
        user = template.format(round=rnd + 1, rounds=g.rounds_per_match, history=self._history(slot), chat=chat)
        self.n_calls += 1
        r = ollama.chat(
            model=self.cfg.model.name,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            options={"temperature": self.cfg.model.temperature, "seed": self.seed + self.n_calls},
        )
        return r["message"]["content"].strip()

    def _history(self, slot: int) -> str:
        me, opp = self.players[slot].history, self.players[1 - slot].history
        if not len(me):
            return "(none yet)"
        lines, total = [], 0.0
        for i, (a, b) in enumerate(zip(me, opp)):
            pts = self.game.score((a, b))[0]
            total += pts
            lines.append(f"Round {i + 1}: you {WORD[a]}, other person {WORD[b]} -> you scored {pts:g}")
        lines.append(f"Your total so far: {total:g}")
        return "\n".join(lines)

    # -- game flow -------------------------------------------------------
    def _talk(self, rnd: int):
        self.chat = []
        first = rnd % 2  # alternate who speaks first to remove first-mover bias
        for _ in range(self.cfg.game.talk_turns_per_round):
            for slot in (first, 1 - first):
                text = self._ask(slot, self.cfg.game.talk_prompt, rnd)
                self.chat.append((slot, text))
                self.emit(ev.Message(self.match_id, rnd, slot, text))

    def decide(self, player: "LLMPlayer", opponent) -> axl.Action:
        rnd, slot = len(player.history), player.slot
        self.players[slot], self.players[1 - slot] = player, opponent
        if self.chat_enabled and self.chatted_round < rnd:
            self.chatted_round = rnd
            self._talk(rnd)
        elif not self.chat_enabled:
            self.chat = []
        raw = self._ask(slot, self.cfg.game.move_prompt, rnd)
        move = parse_move(raw)
        if move is None:  # one retry, then the configured fallback
            raw += " || " + (retry := self._ask(slot, self.cfg.game.move_prompt, rnd))
            move = parse_move(retry) or (C if self.cfg.game.move_fallback == "COOPERATE" else D)
        self.emit(ev.Move(self.match_id, rnd, slot, str(move), raw))
        if self.chat_enabled:
            moves = self.pending.setdefault(rnd, {})
            moves[slot] = move
            if len(moves) == 2:
                pay = self.game.score((moves[0], moves[1]))
                self.emit(ev.Round(self.match_id, rnd, (str(moves[0]), str(moves[1])), (float(pay[0]), float(pay[1]))))
        return move


class LLMPlayer(axl.Player):
    name = "LLM"
    classifier = {
        "memory_depth": float("inf"),
        "stochastic": True,
        "long_run_time": True,
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }

    def __init__(self, room: ChatRoom | None = None, slot: int = 0):
        super().__init__()
        self.room, self.slot = room, slot

    def strategy(self, opponent) -> axl.Action:
        return self.room.decide(self, opponent)


@dataclass
class MatchResult:
    scores: tuple[float, float]
    coop: tuple[float, float]


def play_match(
    match_id: int,
    label: str,
    persona_a: str,
    opponent: "str | axl.Player",
    cfg: Config,
    seed: int,
    emit: Emit = lambda e: None,
    stop: threading.Event | None = None,
) -> MatchResult:
    """`opponent` is persona text (another LLM agent) or an axelrod Player (reference strategy)."""
    g = cfg.game
    opp_is_llm = isinstance(opponent, str)
    room = ChatRoom(cfg, match_id, [persona_a, opponent if opp_is_llm else None], seed, emit, stop)
    pa = LLMPlayer(room, 0)
    pb = LLMPlayer(room, 1) if opp_is_llm else opponent
    units = g.rounds_per_match * (2 * g.talk_turns_per_round + 2) if opp_is_llm else g.rounds_per_match
    emit(ev.MatchStart(match_id, label, persona_a, opponent if opp_is_llm else type(opponent).__name__, units))
    match = axl.Match((pa, pb), turns=g.rounds_per_match, game=room.game, seed=seed)
    match.play()
    s = match.final_score()
    coop = match.cooperation()
    res = MatchResult((float(s[0]), float(s[1])), (coop[0] / g.rounds_per_match, coop[1] / g.rounds_per_match))
    emit(ev.MatchEnd(match_id, res.scores, res.coop))
    return res
