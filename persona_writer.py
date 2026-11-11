"""Standalone persona writer: Big Five vector (-1..1) -> short grounded persona text.

Reads only persona.toml. Usage:
    python persona_writer.py 0.2 -0.5 1 0 -1 [--config persona.toml]
or from code:
    PersonaWriter("persona.toml").describe([0.2, -0.5, 1, 0, -1])
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from pathlib import Path

import ollama

TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]


class PersonaWriter:
    def __init__(self, config_path: str | Path = "persona.toml"):
        path = Path(config_path)
        with open(path, "rb") as f:
            self.cfg = tomllib.load(f)
        self.anchors = {
            t: sorted((float(k), v) for k, v in self.cfg["anchors"][t].items()) for t in TRAITS
        }
        m = self.cfg["model"]
        self.model, self.temperature, self.seed = m["name"], m.get("temperature", 0.0), m.get("seed", 0)
        self.prompt = self.cfg["writer_prompt"]
        # cache is keyed by a hash of everything that influences the output
        self._sig = hashlib.sha256(
            json.dumps([self.model, self.temperature, self.seed, self.prompt, self.cfg["anchors"]], sort_keys=True).encode()
        ).hexdigest()[:12]
        cache = self.cfg.get("cache_file", "")
        self._cache_path = (path.parent / cache) if cache else None
        self._cache: dict[str, str] = {}
        if self._cache_path and self._cache_path.exists():
            try:
                self._cache = json.loads(self._cache_path.read_text())
            except json.JSONDecodeError:
                self._cache = {}

    # -- grounding -------------------------------------------------------
    def _bracket(self, trait: str, v: float):
        pts = self.anchors[trait]
        v = max(pts[0][0], min(pts[-1][0], v))
        for (lo, lo_t), (hi, hi_t) in zip(pts, pts[1:]):
            if lo <= v <= hi:
                return lo, lo_t, hi, hi_t, (v - lo) / (hi - lo)
        raise AssertionError("unreachable")

    def grounding(self, vector) -> str:
        lines = []
        for trait, v in zip(TRAITS, vector):
            lo, lo_t, hi, hi_t, w = self._bracket(trait, float(v))
            head = f"{trait.capitalize()} (position {float(v):+.2f})"
            if w < 0.05:
                lines.append(f'- {head}: "{lo_t}"')
            elif w > 0.95:
                lines.append(f'- {head}: "{hi_t}"')
            else:
                lines.append(
                    f'- {head}: {w:.0%} of the way from position {lo:+.1f} to position {hi:+.1f}.\n'
                    f'    position {lo:+.1f}: "{lo_t}"\n    position {hi:+.1f}: "{hi_t}"'
                )
        return "\n".join(lines)

    # -- public ----------------------------------------------------------
    def describe(self, vector) -> str:
        key = self._sig + ":" + ",".join(f"{float(v):.3f}" for v in vector)
        if key in self._cache:
            return self._cache[key]
        r = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": self.grounding(vector)},
            ],
            options={"temperature": self.temperature, "seed": self.seed},
        )
        text = r["message"]["content"].strip()
        self._cache[key] = text
        if self._cache_path:
            self._cache_path.write_text(json.dumps(self._cache, indent=1))
        return text


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("values", nargs=5, type=float, metavar="O C E A N")
    ap.add_argument("--config", default="persona.toml")
    a = ap.parse_args()
    w = PersonaWriter(a.config)
    print(w.grounding(a.values), "\n")
    print(w.describe(a.values))
