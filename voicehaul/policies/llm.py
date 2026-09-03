"""A real language model as the policy under test.

Everything else in this package is a deterministic simulator, which is what made
the metrics validatable. This is the other half: once the instrument is trusted,
point it at a system nobody controls.

The caller stays simulated and the model under test is real. That is not a
compromise, it is the standard configuration - Hume's own Kairos platform runs
agent-to-agent for scale and human-to-agent for ground truth, and they answer
different questions. A scripted caller is what makes a regression test a test:
if both the caller and the model drift between runs, a difference means nothing.

Responses are cached on disk keyed by (model, prompt, conversation state), so a
re-run costs nothing and produces byte-identical numbers. That matters more than
it sounds: an evaluation you cannot re-run for free is an evaluation nobody
re-runs.
"""

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional

from ..adapters.text import action_from_text
from ..affect import Affect
from ..env import Action, constrain
from .base import VoicePolicy

OLLAMA_ENDPOINT = "http://localhost:11434/api/chat"
HF_ENDPOINT = "https://router.huggingface.co/v1/chat/completions"
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

#: Local first, for the same reason Runopsy is local first: an evaluation you
#: cannot run without someone else's quota is an evaluation you cannot run.
DEFAULT_PROVIDER = "auto"
OLLAMA_MODEL = "qwen2.5:3b"
HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
OPENROUTER_MODEL = "meta-llama/llama-3.1-8b-instruct"

#: OpenAI-compatible providers share a request and response shape.
_OPENAI_SHAPED = ("hf", "openrouter")
CACHE_DIR = os.environ.get("VOICEHAUL_CACHE",
                           os.path.join(".voicehaul-cache", "llm"))


class LLMUnavailable(RuntimeError):
    """Raised when no backend is reachable, so callers can skip cleanly."""


def _ollama_up(endpoint: str = "http://localhost:11434/api/tags") -> bool:
    try:
        with urllib.request.urlopen(endpoint, timeout=2):
            return True
    except Exception:
        return False


ENV_VARS = {
    "hf": ("VOICEHAUL_LLM_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_TOKEN"),
    "openrouter": ("OPENROUTER_API_KEY",),
}


def _token(provider: str = "hf") -> Optional[str]:
    names = ENV_VARS.get(provider, ENV_VARS["hf"])
    for var in names:
        v = os.environ.get(var)
        if v:
            return v.strip()
    import re
    pattern = re.compile(r"\s*({})\s*=\s*(.*)".format("|".join(names)))
    for path in (".env", os.path.join("..", ".env")):
        if os.path.exists(path):
            for line in open(path, encoding="utf-8-sig"):
                m = pattern.match(line)
                if m:
                    return m.group(2).strip().strip('"').strip("'")
    return None


class LLMPolicy(VoicePolicy):
    """Drives a chat model as a voice support agent and measures what it did.

    The model is never asked to rate itself and never sees a metric. It gets a
    system prompt and a conversation, exactly as it would in production; the
    delivery parameters are recovered from its output afterwards.
    """

    def __init__(self, name: str, system_prompt: str,
                 provider: str = DEFAULT_PROVIDER, model: Optional[str] = None,
                 endpoint: Optional[str] = None,
                 max_tokens: int = 160, temperature: float = 0.0,
                 cache_dir: str = CACHE_DIR, timeout: int = 180,
                 max_retries: int = 3):
        if provider == "auto":
            if _token("openrouter"):
                provider = "openrouter"
            elif _ollama_up():
                provider = "ollama"
            else:
                provider = "hf"
        if provider not in ("ollama", "hf", "openrouter"):
            raise ValueError(
                "provider must be 'ollama', 'hf', 'openrouter' or 'auto'")
        self.provider = provider
        self.name = name
        self.system_prompt = system_prompt
        self.model = model or {"ollama": OLLAMA_MODEL, "hf": HF_MODEL,
                               "openrouter": OPENROUTER_MODEL}[provider]
        self.endpoint = endpoint or {"ollama": OLLAMA_ENDPOINT,
                                     "hf": HF_ENDPOINT,
                                     "openrouter": OPENROUTER_ENDPOINT}[provider]
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.max_retries = max_retries
        self.calls = 0
        self.cache_hits = 0
        self.transcript: List[Dict[str, str]] = []

    # -- lifecycle ----------------------------------------------------------

    def reset(self, seed: int) -> None:
        super().reset(seed)
        self.transcript = []
        self._pending_caller: Optional[str] = None
        self._last_reply = ""

    def hear(self, utterance: str) -> None:
        """The caller speaks. Called by the runner before act()."""
        self._pending_caller = utterance

    # -- the model call -----------------------------------------------------

    def _cache_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, key[:2], key + ".json")

    def _payload(self, messages: List[Dict[str, str]]) -> Dict:
        if self.provider == "ollama":
            return {"model": self.model, "messages": messages, "stream": False,
                    "options": {"temperature": self.temperature,
                                "num_predict": self.max_tokens, "seed": 0}}
        return {"model": self.model, "messages": messages,
                "max_tokens": self.max_tokens, "temperature": self.temperature}

    def _extract(self, data: Dict) -> str:
        if self.provider == "ollama":
            return data["message"]["content"].strip()
        return data["choices"][0]["message"]["content"].strip()

    def _complete(self, messages: List[Dict[str, str]]) -> str:
        payload = self._payload(messages)
        key = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()).hexdigest()[:32]
        path = self._cache_path(key)
        if os.path.exists(path):
            self.cache_hits += 1
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)["content"]

        headers = {"Content-Type": "application/json", "User-Agent": "voicehaul"}
        if self.provider in _OPENAI_SHAPED:
            token = _token(self.provider)
            if not token:
                raise LLMUnavailable(
                    "no credential for provider {!r}; set one of {}, or run a "
                    "local model with ollama".format(
                        self.provider, ", ".join(ENV_VARS[self.provider])))
            headers["Authorization"] = "Bearer " + token

        last = None
        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(
                    self.endpoint, data=json.dumps(payload).encode(),
                    headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = json.load(r)
                content = self._extract(data)
                self.calls += 1
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump({"content": content, "model": self.model,
                               "usage": data.get("usage")}, fh)
                return content
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "replace")[:200]
                if e.code in (401, 402, 403):     # no retry will fix a quota
                    raise LLMUnavailable("{} refused the call: HTTP {} {}".format(
                        self.provider, e.code, body))
                last = "HTTP {} {}".format(e.code, body)
                time.sleep(1.5 * (attempt + 1))
            except (urllib.error.URLError, OSError) as e:
                last = e
                time.sleep(1.5 * (attempt + 1))
        raise LLMUnavailable("model call failed after {} attempts: {}".format(
            self.max_retries, last))

    # -- policy interface ---------------------------------------------------

    def act(self, user: Affect, turn: int) -> Action:
        caller = self._pending_caller or "..."
        self.transcript.append({"role": "user", "content": caller})
        messages = ([{"role": "system", "content": self.system_prompt}]
                    + self.transcript[-12:])
        reply = self._complete(messages)
        self.transcript.append({"role": "assistant", "content": reply})
        self._last_reply = reply

        caller_energy = max(0.0, min(1.0, 0.5 + 0.5 * user.arousal))
        action = action_from_text(reply, caller_energy)
        # Standing requests are honoured only if the model actually honoured
        # them. constrain() is not applied: that would grant compliance the
        # model did not earn, and uptake is the metric under test.
        return action

    def observe_directive(self, directive: str) -> None:
        # Intentionally a no-op. A real model has to pick the request up from
        # the conversation; injecting it out of band would measure nothing.
        pass


# ---------------------------------------------------------------------------
# the two prompts a voice product team actually ships between
# ---------------------------------------------------------------------------

WARM_MIRRORING = (
    "You are a voice support agent for a utilities company. Speak naturally, as "
    "if on a phone call. Be warm, enthusiastic and empathetic. Match the "
    "caller's energy so they feel heard, and mirror their tone. Keep the "
    "conversation upbeat and reassuring. Reply with one or two spoken "
    "sentences only."
)

CALM_REGULATING = (
    "You are a voice support agent for a utilities company. Speak naturally, as "
    "if on a phone call. Stay a little calmer and slower than the caller: never "
    "exceed their energy, and let your steadiness give them something to settle "
    "against. Acknowledge what they have told you before you add anything new. "
    "If they ask you to change how you speak, keep doing it for the rest of the "
    "call. Reply with one or two spoken sentences only."
)


def warm_mirroring(**kw) -> LLMPolicy:
    return LLMPolicy("llm-warm-mirroring", WARM_MIRRORING, **kw)


def calm_regulating(**kw) -> LLMPolicy:
    return LLMPolicy("llm-calm-regulating", CALM_REGULATING, **kw)
