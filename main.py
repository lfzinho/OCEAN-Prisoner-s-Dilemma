"""Run the OCEAN x Prisoner's Dilemma search.  python main.py [config.toml] [--headless]"""
from __future__ import annotations

import argparse
import json
import shutil
import threading
from datetime import datetime
from pathlib import Path

from persona_writer import PersonaWriter
from pd_search import events as ev
from pd_search.config import load_config
from pd_search.optimizer import Optimizer


def headless_print(e) -> None:
    if isinstance(e, ev.MatchStart):
        print(f"\n== match {e.match_id}: {e.label}")
    elif isinstance(e, ev.Message):
        print(f"  [{'AB'[e.slot]} r{e.round + 1}] {e.text}")
    elif isinstance(e, ev.Round):
        print(f"  round {e.round + 1}: {e.moves} -> {e.payoffs}")
    elif isinstance(e, ev.MatchEnd):
        print(f"  scores {e.scores}")
    elif isinstance(e, ev.StepEnd):
        print(f"** step {e.step} agent {'AB'[e.player]}: {e.baseline:g} -> {e.best:g} "
              f"{'ACCEPT' if e.accepted else 'reject'} lr={e.lr:.4f}")
    elif isinstance(e, ev.Final):
        print(json.dumps(e.summary, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="config.toml")
    ap.add_argument("--headless", action="store_true", help="plain log instead of the TUI")
    args = ap.parse_args()

    cfg = load_config(args.config)
    persona = PersonaWriter(cfg.persona_path())
    run_dir = Path("runs") / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)
    shutil.copy(args.config, run_dir / "config.toml")
    shutil.copy(cfg.persona_path(), run_dir / "persona.toml")
    log = open(run_dir / "events.jsonl", "a", buffering=1)

    def work(ui_emit, stop: threading.Event) -> None:
        def emit(e) -> None:
            log.write(json.dumps(ev.to_dict(e)) + "\n")
            if isinstance(e, ev.Final):
                (run_dir / "summary.json").write_text(json.dumps(e.summary, indent=2))
            ui_emit(e)
        Optimizer(cfg, persona, emit, stop).run()

    if args.headless:
        work(headless_print, threading.Event())
    else:
        from pd_search.tui import PDApp
        PDApp(cfg, work).run()
    print(f"run saved in {run_dir}")


if __name__ == "__main__":
    main()
