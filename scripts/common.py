"""Shared utilities for the pipeline scripts.

Centralizes:
  * Settings loading
  * Anthropic client construction and retry/backoff
  * Token + cost tracking (prompt-cache aware)
  * Per-call audit logging
  * Robust JSON parsing
  * PDF text extraction
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import yaml

try:
    import anthropic
except ImportError:
    anthropic = None  # imported lazily; orchestrator will surface a friendly error


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "settings.yaml"
LOG_DIR = REPO_ROOT / "logs"
OUTPUTS_DIR = REPO_ROOT / "outputs"


def load_settings(path: Path | None = None) -> dict[str, Any]:
    path = path or CONFIG_PATH
    with open(path) as f:
        return yaml.safe_load(f)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(h)
    return logger


# ---------------------------------------------------------------------------
# Anthropic client + retry
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int
    model: str
    stop_reason: str | None
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class CostTracker:
    """Aggregates token usage and dollar cost across a run.

    Accounts for prompt caching: cache writes cost 1.25× base input, cache reads cost 0.1× base input.
    """

    def __init__(self, pricing: dict[str, dict[str, float]]):
        self.pricing = pricing
        self.input_tokens: dict[str, int] = {}          # uncached input (regular)
        self.output_tokens: dict[str, int] = {}
        self.cache_write_tokens: dict[str, int] = {}    # cache creation
        self.cache_read_tokens: dict[str, int] = {}     # cache hits

    def add(self, model: str, in_t: int, out_t: int, cache_write: int = 0, cache_read: int = 0) -> None:
        # `in_t` from the API already excludes cached tokens; track each bucket separately.
        self.input_tokens[model] = self.input_tokens.get(model, 0) + in_t
        self.output_tokens[model] = self.output_tokens.get(model, 0) + out_t
        if cache_write:
            self.cache_write_tokens[model] = self.cache_write_tokens.get(model, 0) + cache_write
        if cache_read:
            self.cache_read_tokens[model] = self.cache_read_tokens.get(model, 0) + cache_read

    def total_cost_usd(self) -> float:
        total = 0.0
        all_models = set(self.input_tokens) | set(self.output_tokens) | set(self.cache_write_tokens) | set(self.cache_read_tokens)
        for model in all_models:
            p = self.pricing.get(model)
            if not p:
                continue
            total += self.input_tokens.get(model, 0) * p["input"] / 1_000_000
            total += self.output_tokens.get(model, 0) * p["output"] / 1_000_000
            total += self.cache_write_tokens.get(model, 0) * p["input"] * 1.25 / 1_000_000
            total += self.cache_read_tokens.get(model, 0) * p["input"] * 0.10 / 1_000_000
        return total

    def summary(self) -> str:
        rows = []
        all_models = set(self.input_tokens) | set(self.output_tokens) | set(self.cache_write_tokens) | set(self.cache_read_tokens)
        for model in sorted(all_models):
            cw = self.cache_write_tokens.get(model, 0)
            cr = self.cache_read_tokens.get(model, 0)
            extra = f" cache_write={cw:,} cache_read={cr:,}" if (cw or cr) else ""
            rows.append(
                f"  {model}: in={self.input_tokens.get(model,0):,} out={self.output_tokens.get(model,0):,}{extra}"
            )
        return "Tokens:\n" + "\n".join(rows) + f"\nTotal cost: ${self.total_cost_usd():.4f}"


def get_client():
    if anthropic is None:
        raise RuntimeError("anthropic SDK not installed. `pip install anthropic pyyaml pypdf`.")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in environment.")
    return anthropic.Anthropic(api_key=api_key)


def call_llm(
    *,
    client,
    model: str,
    system: str | list[dict],
    user: str,
    max_tokens: int,
    temperature: float,
    retry_cfg: dict[str, Any],
    log_path: Path | None = None,
    stage: str | None = None,
    paper_id: str | None = None,
) -> LLMResult:
    """Call the Anthropic Messages API with retry/backoff. Logs prompt + response if log_path set.

    `system` may be a plain string OR a list of content blocks (for prompt caching), e.g.:
        [{"type": "text", "text": "..."},
         {"type": "text", "text": "<large cacheable payload>",
          "cache_control": {"type": "ephemeral"}}]
    """
    attempts = retry_cfg.get("max_attempts", 5)
    backoff = float(retry_cfg.get("initial_backoff_seconds", 2.0))
    max_backoff = float(retry_cfg.get("max_backoff_seconds", 60.0))

    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            ) as stream:
                resp = stream.get_final_message()
            text_parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
            text = "\n".join(text_parts)
            result = LLMResult(
                text=text,
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
                model=model,
                stop_reason=getattr(resp, "stop_reason", None),
                cache_creation_input_tokens=getattr(resp.usage, "cache_creation_input_tokens", 0) or 0,
                cache_read_input_tokens=getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
            )
            if log_path is not None:
                _append_audit_log(
                    log_path,
                    {
                        "ts": time.time(),
                        "stage": stage,
                        "paper_id": paper_id,
                        "model": model,
                        "temperature": temperature,
                        "system": system if isinstance(system, str) else "<structured>",
                        "user": user,
                        "response_text": text,
                        "input_tokens": result.input_tokens,
                        "output_tokens": result.output_tokens,
                        "cache_creation_input_tokens": result.cache_creation_input_tokens,
                        "cache_read_input_tokens": result.cache_read_input_tokens,
                        "stop_reason": result.stop_reason,
                    },
                )
            return result
        except Exception as e:  # noqa: BLE001 — retry anything transient
            last_exc = e
            if attempt == attempts:
                break
            sleep = min(max_backoff, backoff * (2 ** (attempt - 1))) * (0.5 + random.random())
            time.sleep(sleep)
    assert last_exc is not None
    raise last_exc


def _append_audit_log(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_json_response(text: str) -> Any:
    """Parse JSON from an LLM response that may be wrapped in fences or prose.

    Strategy (each tried only if the previous one raises):
      1. Direct json.loads of stripped text.
      2. If wrapped in ```...```, strip the outermost fence and parse the body.
      3. Non-greedy regex extraction of a ```json ... ``` block.
      4. Outer-brace substring (handles prose before/after the object).
      5. Outer-bracket substring (array root).
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # (2) Fenced: trim outermost ``` markers, regardless of nested code in content.
    if text.startswith("```"):
        body = text
        nl = body.find("\n")
        body = body[nl + 1 :] if nl != -1 else body[3:]
        if body.rstrip().endswith("```"):
            body = body.rstrip()[:-3].rstrip()
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            pass

    # (3) Regex fenced block (works when the JSON is embedded mid-response).
    m = _JSON_BLOCK_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    # (4) Outer braces
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError as e:
            raise ValueError(
                f"All JSON parse strategies failed. Final error: {e}. "
                f"First 200 chars: {text[:200]!r}"
            ) from e

    # (5) Outer brackets (array root)
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])

    raise ValueError(f"Could not parse JSON from response:\n{text[:500]}")


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from a PDF using pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise RuntimeError("pypdf not installed. `pip install pypdf`") from e
    reader = PdfReader(str(pdf_path))
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:  # noqa: BLE001
            parts.append("")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Paper IDs and JSON paths
# ---------------------------------------------------------------------------

def paper_id_from_path(pdf_path: Path) -> str:
    return pdf_path.stem


def load_prompt(name: str) -> str:
    return (REPO_ROOT / "prompts" / name).read_text()


def output_path(stage_dir: str, paper_id: str) -> Path:
    return OUTPUTS_DIR / stage_dir / f"{paper_id}.json"


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def read_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def stage_done(stage_dir: str, paper_id: str) -> bool:
    return output_path(stage_dir, paper_id).exists()
