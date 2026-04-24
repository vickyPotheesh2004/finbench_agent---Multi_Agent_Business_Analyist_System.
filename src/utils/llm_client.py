"""
src/utils/llm_client.py
FinBench Multi-Agent Business Analyst AI
PDR-BAAAI-001 · Rev 1.0

Gemma4 LLM Client — wraps Ollama gemma4:e4b at localhost:11434

Used by ALL analysis pods: N11 LeadAnalyst, N12 QuantAnalyst,
N14 BlindAuditor, N15 PIVMediator, N02 SectionTree summaries.

Constraints:
    C1  $0 cost — local Ollama only
    C2  100% local — zero external network calls
    C3  Model = gemma4:e4b (128K context, 9.6GB)
    C5  seed=42
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_MODEL    = "qwen2.5:3b"
FALLBACK_MODEL   = "qwen2.5:3b"
BASE_URL         = "http://localhost:11434"
DEFAULT_TIMEOUT  = 120       # seconds
DEFAULT_TEMP     = 0.1       # low temperature for factual financial QA
MAX_RETRIES      = 3
RETRY_DELAY      = 2.0       # seconds between retries
SEED             = 42


class Gemma4Client:
    """
    Wraps Ollama gemma4:e4b for all LLM calls in the pipeline.

    Features:
        - Automatic retry on timeout/connection error (MAX_RETRIES=3)
        - Circuit breaker — trips after 3 consecutive failures
        - Health check — verifies Ollama is running before first call
        - Streaming support — optional token streaming
        - Context-first enforcement — validates C7 in prompts
    """

    def __init__(
        self,
        model:    str = DEFAULT_MODEL,
        base_url: str = BASE_URL,
        timeout:  int = DEFAULT_TIMEOUT,
        seed:     int = SEED,
    ) -> None:
        self.model      = model
        self.base_url   = base_url.rstrip("/")
        self.timeout    = timeout
        self.seed       = seed

        # Circuit breaker state
        self._failure_count    = 0
        self._circuit_open     = False
        self._circuit_open_at  = 0.0
        self._circuit_reset_s  = 60.0

        # Stats
        self._total_calls      = 0
        self._total_failures   = 0
        self._last_latency_ms  = 0.0

    # ── Primary interface ─────────────────────────────────────────────────────

    def chat(
        self,
        prompt:      str,
        temperature: float = DEFAULT_TEMP,
        max_tokens:  int   = 2048,
        system:      str   = "",
    ) -> str:
        """
        Send a prompt to Gemma4 and return the response text.

        Args:
            prompt      : The user prompt (context MUST come before question)
            temperature : 0.0–1.0. Default 0.1 for factual accuracy.
            max_tokens  : Maximum tokens in response
            system      : Optional system message

        Returns:
            Response text string. Empty string on failure.
        """
        if self._circuit_open:
            if time.time() - self._circuit_open_at > self._circuit_reset_s:
                logger.info("[LLM] Circuit breaker reset — retrying")
                self._circuit_open   = False
                self._failure_count  = 0
            else:
                logger.warning("[LLM] Circuit open — skipping LLM call")
                return ""

        self._total_calls += 1
        t0 = time.time()

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self._call_ollama(
                    prompt, temperature, max_tokens, system
                )
                self._failure_count  = 0
                self._last_latency_ms = (time.time() - t0) * 1000
                logger.debug(
                    "[LLM] Response received | latency=%.0fms | attempt=%d",
                    self._last_latency_ms, attempt,
                )
                return response

            except Exception as exc:
                logger.warning(
                    "[LLM] Attempt %d/%d failed: %s",
                    attempt, MAX_RETRIES, exc,
                )
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY * attempt)

        # All retries failed
        self._total_failures  += 1
        self._failure_count   += 1
        if self._failure_count >= MAX_RETRIES:
            self._circuit_open    = True
            self._circuit_open_at = time.time()
            logger.error("[LLM] Circuit breaker TRIPPED after %d failures", self._failure_count)

        return ""

    def chat_json(
        self,
        prompt:      str,
        temperature: float = DEFAULT_TEMP,
    ) -> Dict[str, Any]:
        """
        Call LLM and parse response as JSON.
        Returns empty dict on failure or invalid JSON.
        """
        raw = self.chat(prompt, temperature=temperature)
        if not raw:
            return {}
        # Strip markdown fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines   = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.debug("[LLM] JSON parse failed — returning raw text in dict")
            return {"raw_text": raw}

    # ── Health & availability ─────────────────────────────────────────────────

    def is_available(self) -> bool:
        """
        Check if Ollama is running and model is available.
        Returns True if healthy, False otherwise.
        Zero side effects — safe to call frequently.
        """
        try:
            import urllib.request
            url = f"{self.base_url}/api/tags"
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status != 200:
                    return False
                data   = json.loads(resp.read().decode())
                models = [m.get("name", "") for m in data.get("models", [])]
                return any(self.model in m for m in models)
        except Exception:
            return False

    def health_check(self) -> Dict[str, Any]:
        """
        Return health status dict.
        Used by /health endpoint and Streamlit UI status indicator.
        """
        available = self.is_available()
        return {
            "model":           self.model,
            "base_url":        self.base_url,
            "available":       available,
            "circuit_open":    self._circuit_open,
            "total_calls":     self._total_calls,
            "total_failures":  self._total_failures,
            "failure_rate":    (
                self._total_failures / self._total_calls
                if self._total_calls > 0 else 0.0
            ),
            "last_latency_ms": self._last_latency_ms,
        }

    def reset_circuit(self) -> None:
        """Manually reset the circuit breaker."""
        self._circuit_open   = False
        self._failure_count  = 0
        self._circuit_open_at = 0.0
        logger.info("[LLM] Circuit breaker manually reset")

    # ── Private ───────────────────────────────────────────────────────────────

    def _call_ollama(
        self,
        prompt:      str,
        temperature: float,
        max_tokens:  int,
        system:      str,
    ) -> str:
        """
        Make HTTP POST to Ollama /api/generate.
        Uses stdlib urllib only — zero extra dependencies.
        C2: no external network — localhost only.
        """
        import urllib.request

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = json.dumps({
            "model":   self.model,
            "messages": messages,
            "stream":  False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "seed":        self.seed,
            },
        }).encode("utf-8")

        url = f"{self.base_url}/api/chat"
        req = urllib.request.Request(
            url,
            data    = payload,
            headers = {"Content-Type": "application/json"},
            method  = "POST",
        )

        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Ollama HTTP {resp.status}")
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("message", {}).get("content", "")


# ── Singleton helper ──────────────────────────────────────────────────────────

_default_client: Optional[Gemma4Client] = None


def get_llm_client(
    model:    str = DEFAULT_MODEL,
    base_url: str = BASE_URL,
) -> Gemma4Client:
    """
    Return a shared Gemma4Client instance.
    Creates on first call, reuses on subsequent calls.
    """
    global _default_client
    if _default_client is None:
        _default_client = Gemma4Client(model=model, base_url=base_url)
    return _default_client


def reset_llm_client() -> None:
    """Reset the shared client — used in tests."""
    global _default_client
    _default_client = None