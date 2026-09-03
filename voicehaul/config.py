"""Suite configuration.

A run is defined by a config, and a config hashes to a short id that is stamped
onto every artifact it produces. Two reports carrying the same suite id were
measured the same way; two carrying different ids were not, and comparing them
is a mistake the report can refuse to make.
"""

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

DEFAULT_PERSONAS = ["distressed_billing", "hostile_escalation", "confused_elderly",
                    "grieving_claim", "cautious_optimist"]


@dataclass(frozen=True)
class SuiteConfig:
    """Everything that changes a number, and nothing that does not."""

    name: str = "default"
    episodes: int = 30
    turns: int = 40
    seed: int = 0
    personas: tuple = tuple(DEFAULT_PERSONAS)
    corrupt_p: float = 0.0
    #: probability a directive reaches the model as its opposite (ASR error)
    rater_sigma: float = 0.9
    #: per-rater noise in Likert points, used by the power model
    raters_per_conversation: int = 3
    tolerance: int = 1
    #: how close an onset prediction must be to count as correct

    def __post_init__(self):
        if self.episodes < 1:
            raise ValueError("episodes must be >= 1")
        if self.turns < 5:
            raise ValueError("turns must be >= 5; nothing long-horizon happens below that")
        if not 0.0 <= self.corrupt_p <= 1.0:
            raise ValueError("corrupt_p must be a probability")
        if not self.personas:
            raise ValueError("a suite needs at least one persona")
        object.__setattr__(self, "personas", tuple(self.personas))

    # -- provenance ---------------------------------------------------------

    @property
    def suite_id(self) -> str:
        """Short stable hash of everything that affects a number."""
        payload = json.dumps(asdict(self), sort_keys=True, default=list)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["personas"] = list(self.personas)
        d["suite_id"] = self.suite_id
        return d

    # -- io -----------------------------------------------------------------

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SuiteConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def load(cls, path: str) -> "SuiteConfig":
        """Load JSON, or a minimal flat YAML if the file ends in .yaml/.yml.

        The YAML reader handles `key: value` and `key: [a, b]` only. That covers
        every config this package needs, and it keeps PyYAML off the dependency
        list for a benchmark whose whole point is that it runs anywhere.
        """
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        if os.path.splitext(path)[1].lower() in (".yaml", ".yml"):
            return cls.from_dict(_parse_flat_yaml(text))
        return cls.from_dict(json.loads(text))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    def replace(self, **kw) -> "SuiteConfig":
        d = asdict(self)
        d.update(kw)
        return SuiteConfig.from_dict(d)

    def describe(self) -> str:
        return ("{name}  [{sid}]  {n} conversations x {t} turns x {p} callers"
                "  seed={s}{c}").format(
            name=self.name, sid=self.suite_id, n=self.episodes, t=self.turns,
            p=len(self.personas), s=self.seed,
            c="  corrupt={:.0%}".format(self.corrupt_p) if self.corrupt_p else "")


def _parse_flat_yaml(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, _, val = line.partition(":")
        key, val = key.strip(), val.strip()
        if not val:
            continue
        if val.startswith("[") and val.endswith("]"):
            items = [v.strip().strip("'\"") for v in val[1:-1].split(",")]
            out[key] = [v for v in items if v]
        elif val.lower() in ("true", "false"):
            out[key] = val.lower() == "true"
        else:
            try:
                out[key] = int(val)
            except ValueError:
                try:
                    out[key] = float(val)
                except ValueError:
                    out[key] = val.strip("'\"")
    return out
