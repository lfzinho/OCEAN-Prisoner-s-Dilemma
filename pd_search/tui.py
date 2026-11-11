"""Textual TUI: progress dashboard, live/browsable match chats, current personas."""
from __future__ import annotations

import math
import threading
from typing import Callable

from rich.markup import escape
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Label, ProgressBar, RichLog, Static, TabbedContent, TabPane

from . import events as ev
from .config import Config

TRAITS = ["O", "C", "E", "A", "N"]
COLORS = ["cyan", "magenta"]
NAMES = ["A", "B"]


def trait_bar(v: float, lo: float, hi: float, width: int = 25) -> str:
    frac = (v - lo) / (hi - lo)
    pos = max(0, min(width - 1, round(frac * (width - 1))))
    mid = (width - 1) // 2
    cells = ["─"] * width
    cells[mid] = "┼"
    for i in range(min(mid, pos), max(mid, pos) + 1):
        cells[i] = "█" if i != mid else "┼"
    cells[pos] = "●"
    return "".join(cells)


class PDApp(App):
    TITLE = "OCEAN × Prisoner's Dilemma"
    CSS = """
    .agent { border: round $primary; padding: 0 1; height: auto; width: 1fr; }
    #global, #matchinfo { padding: 0 1; height: auto; }
    #steplog { height: 1fr; border: round $surface; }
    #chatlog { height: 1fr; }
    #chathead { padding: 0 1; background: $boost; }
    """
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("d", "tab('dash')", "Dashboard"),
        ("c", "tab('chat')", "Chat"),
        ("p", "tab('pers')", "Personas"),
        Binding("left", "browse(-1)", "Prev match", priority=True),
        Binding("right", "browse(1)", "Next match", priority=True),
        ("l", "live", "Live"),
    ]

    def __init__(self, cfg: Config, work: Callable[[Callable, threading.Event], None]):
        super().__init__()
        self.cfg, self.work = cfg, work
        self.stop = threading.Event()
        self.matches: dict[int, dict] = {}
        self.order: list[int] = []
        self.view: int | None = None  # None = follow live
        self.current: int | None = None
        self.units_done = 0
        self.units_total = 1
        self.state: ev.State | None = None
        self.final: dict | None = None

    # -- layout ----------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(initial="dash", id="tabs"):
            with TabPane("Dashboard", id="dash"):
                yield Static("", id="global")
                with Horizontal():
                    for i in range(2):
                        with Vertical(classes="agent"):
                            yield Static("", id=f"title{i}")
                            yield Static("", id=f"traits{i}")
                            yield Label("learning rate → min (log scale)")
                            yield ProgressBar(total=100, show_eta=False, id=f"lr{i}")
                yield Static("", id="matchinfo")
                yield ProgressBar(total=100, show_eta=False, id="matchbar")
                yield RichLog(id="steplog", markup=True, wrap=True)
            with TabPane("Chat", id="chat"):
                yield Static("", id="chathead")
                yield RichLog(id="chatlog", markup=True, wrap=True)
            with TabPane("Personas", id="pers"):
                yield Static("", id="personas")
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = "running"
        self._refresh_dash()
        threading.Thread(target=self._run_work, daemon=True).start()

    def _run_work(self):
        def emit(e):
            try:
                self.call_from_thread(self.handle, e)
            except Exception:  # app already closed
                pass
        try:
            self.work(emit, self.stop)
        except Exception as exc:  # surface crashes in the UI
            try:
                self.call_from_thread(self._crash, exc)
            except Exception:
                pass

    def _crash(self, exc):
        self.sub_title = "crashed"
        self.query_one("#steplog", RichLog).write(f"[bold red]ERROR: {escape(repr(exc))}[/]")

    async def action_quit(self) -> None:
        self.stop.set()
        self.exit()

    # -- actions ---------------------------------------------------------
    def action_tab(self, name: str) -> None:
        self.query_one("#tabs", TabbedContent).active = name

    def action_browse(self, d: int) -> None:
        if not self.order:
            return
        cur = self.view if self.view is not None else self.order[-1]
        idx = max(0, min(len(self.order) - 1, self.order.index(cur) + d))
        self.view = self.order[idx]
        if self.view == self.order[-1] and self.view == self.current:
            self.view = None
        self._render_chat()

    def action_live(self) -> None:
        self.view = None
        self._render_chat()

    # -- events ----------------------------------------------------------
    def handle(self, e) -> None:
        if isinstance(e, ev.MatchStart):
            self.matches[e.match_id] = {"label": e.label, "lines": [], "a": e.persona_a, "b": e.persona_b}
            self.order.append(e.match_id)
            self.current = e.match_id
            self.units_done, self.units_total = 0, max(1, e.total_units)
            self._line(e.match_id, f"[dim]— {escape(e.label)} —[/]")
            self.query_one("#personas", Static).update(
                f"[bold {COLORS[0]}]Agent A[/]\n{escape(e.persona_a)}\n\n"
                f"[bold {COLORS[1]}]Agent B / opponent[/]\n{escape(e.persona_b)}"
            )
        elif isinstance(e, ev.Message):
            self.units_done += 1
            self._line(e.match_id, f"[bold {COLORS[e.slot]}]{NAMES[e.slot]}[/] [dim]r{e.round + 1}[/]: {escape(e.text)}")
        elif isinstance(e, ev.Move):
            self.units_done += 1
        elif isinstance(e, ev.Round):
            m = {"C": "[green]COOPERATE[/]", "D": "[red]DEFECT[/]"}
            self._line(e.match_id, f"[bold]Round {e.round + 1}:[/] A {m[e.moves[0]]}  B {m[e.moves[1]]}  → "
                                   f"A +{e.payoffs[0]:g}  B +{e.payoffs[1]:g}")
        elif isinstance(e, ev.MatchEnd):
            self._line(e.match_id, f"[bold]Match over:[/] A {e.scores[0]:g}  B {e.scores[1]:g}  "
                                   f"(coop A {e.coop[0]:.0%}, B {e.coop[1]:.0%})")
            self.units_done = self.units_total
        elif isinstance(e, ev.StepEnd):
            verdict = "[green]accepted[/]" if e.accepted else "[yellow]rejected, lr halved[/]"
            self.query_one("#steplog", RichLog).write(
                f"step {e.step} · agent {NAMES[e.player]} · baseline {e.baseline:g} → best {e.best:g} · {verdict} · lr {e.lr:.4f}")
        elif isinstance(e, ev.State):
            self.state = e
        elif isinstance(e, ev.Final):
            self.final = e.summary
            self.sub_title = "finished"
            self._write_summary(e.summary)
        self._refresh_dash()

    def _line(self, match_id: int, line: str) -> None:
        self.matches[match_id]["lines"].append(line)
        if (self.view if self.view is not None else self.current) == match_id:
            self.query_one("#chatlog", RichLog).write(line)
            self._chat_head()

    def _chat_head(self) -> None:
        mid = self.view if self.view is not None else self.current
        if mid is None:
            return
        tag = "[green]LIVE[/]" if self.view is None else "[yellow]browsing (l = back to live)[/]"
        pos = self.order.index(mid) + 1
        self.query_one("#chathead", Static).update(
            f"Match {pos}/{len(self.order)} · {escape(self.matches[mid]['label'])} · {tag} · ←/→ browse")

    def _render_chat(self) -> None:
        log = self.query_one("#chatlog", RichLog)
        log.clear()
        mid = self.view if self.view is not None else self.current
        if mid is not None:
            for line in self.matches[mid]["lines"]:
                log.write(line)
        self._chat_head()

    def _write_summary(self, s: dict) -> None:
        log = self.query_one("#steplog", RichLog)
        log.write("[bold green]=== FINAL RESULTS ===[/]")
        for tag, a in s["agents"].items():
            vec = "  ".join(f"{t}={v:+.2f}" for t, v in zip(TRAITS, a["vector"]))
            log.write(f"Agent {tag}: {a['mean_score']:.1f} ± {a['std_score']:.1f}  ({vec})")
        for k, v in s["reference_scores"].items():
            log.write(f"  {k}: {v['mean']:.1f} ± {v['std']:.1f}")

    # -- dashboard -------------------------------------------------------
    def _lr_progress(self, lr: float) -> float:
        s = self.cfg.search
        span = math.log(s.initial_lr / s.min_lr)
        if span <= 0:
            return 100.0
        return max(0.0, min(100.0, 100 * math.log(s.initial_lr / lr) / span))

    def _refresh_dash(self) -> None:
        st, s = self.state, self.cfg.search
        if st is None:
            self.query_one("#global", Static).update("waiting for first match…")
            return
        remaining = [max(0, math.ceil(math.log(l / s.min_lr * 1.0000001, 2))) if l >= s.min_lr else 0 for l in st.lr]
        self.query_one("#global", Static).update(
            f"[bold]Matches played:[/] {st.games}   [bold]Step:[/] {st.step}   [bold]Phase:[/] {escape(st.phase)}\n"
            f"[bold]Rejections still needed (A / B) before lr < min:[/] {remaining[0]} / {remaining[1]}")
        for i in range(2):
            done = st.lr[i] < s.min_lr
            self.query_one(f"#title{i}", Static).update(
                f"[bold {COLORS[i]}]Agent {NAMES[i]}[/]  lr {st.lr[i]:.4f} / min {s.min_lr}"
                f"{'  [green]done[/]' if done else ''}\n"
                f"accepted moves {st.accepted[i]} · rejected {st.rejected[i]}")
            rows = [f"{t} {v:+.2f} {trait_bar(v, s.trait_min, s.trait_max)}" for t, v in zip(TRAITS, st.x[i])]
            self.query_one(f"#traits{i}", Static).update("\n".join(rows))
            self.query_one(f"#lr{i}", ProgressBar).update(progress=self._lr_progress(st.lr[i]))
        pct = 100 * self.units_done / self.units_total
        mid = self.current
        label = self.matches[mid]["label"] if mid else "-"
        self.query_one("#matchinfo", Static).update(f"[bold]Current match:[/] {escape(label)} — {pct:.0f}%")
        self.query_one("#matchbar", ProgressBar).update(progress=pct)
