"""
src/live_data/transcripts.py
FinBench Multi-Agent Business Analyst AI
PDR-BAAAI-001 · Rev 1.0

Phase 7C — Earnings Call Transcripts

Parses and analyses earnings call transcripts for sentiment,
forward guidance extraction, and management tone analysis.
Feeds into InvestorRelationsAgent (post-launch) for IR-focused queries.

C2 NOTE: Network only if TRANSCRIPTS_ENABLED=True.
         Parsing and sentiment analysis run 100% local.

Constraints:
    C1  $0 cost — BeautifulSoup + rule-based sentiment
    C2  Network ONLY for fetch — parsing stays local
    C5  seed=42
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

TRANSCRIPTS_ENABLED = os.getenv("TRANSCRIPTS_ENABLED", "false").lower() == "true"
DEFAULT_CACHE_DIR   = "data/transcripts_cache"
SEED                = 42

# Rule-based sentiment lexicons (no ML, no network)
POSITIVE_WORDS = {
    "strong", "record", "beat", "exceeded", "outperform", "growth",
    "robust", "solid", "excellent", "momentum", "accelerated",
    "improving", "expanding", "raised", "upgrade", "bullish",
    "confident", "optimistic", "healthy", "favorable", "positive",
    "increased", "higher", "grew", "growing", "improved",
    "milestone", "leadership", "innovation", "breakthrough",
}

NEGATIVE_WORDS = {
    "weak", "decline", "declined", "missed", "below", "disappointing",
    "headwinds", "challenging", "pressure", "slowdown", "softness",
    "lower", "decreased", "impacted", "uncertain", "cautious",
    "concerned", "difficult", "challenges", "volatile", "reduced",
    "loss", "losses", "deteriorated", "weakness", "macro",
    "restructuring", "layoffs", "impairment",
}

GUIDANCE_KEYWORDS = {
    "guidance", "outlook", "expect", "expects", "expecting",
    "forecast", "anticipate", "project", "projected",
    "target", "targets", "estimates", "range",
    "next quarter", "next year", "full year",
    "fiscal 202", "fy202",
}

RISK_KEYWORDS = {
    "risk", "risks", "uncertainty", "uncertain",
    "volatility", "headwind", "pressure",
    "macro environment", "foreign exchange", "fx",
    "supply chain", "inflation", "recession",
}


class TranscriptSegment:
    """A speaker segment from an earnings call."""

    __slots__ = ("speaker", "role", "text", "position")

    def __init__(
        self,
        speaker:  str,
        text:     str,
        role:     str = "Unknown",
        position: int = 0,
    ) -> None:
        self.speaker  = speaker
        self.role     = role
        self.text     = text
        self.position = position

    def to_dict(self) -> Dict:
        return {
            "speaker":  self.speaker,
            "role":     self.role,
            "text":     self.text,
            "position": self.position,
        }


class SentimentScore:
    """Sentiment analysis result for a transcript or segment."""

    def __init__(
        self,
        positive_count: int   = 0,
        negative_count: int   = 0,
        total_words:    int   = 0,
    ) -> None:
        self.positive_count = positive_count
        self.negative_count = negative_count
        self.total_words    = max(total_words, 1)

    @property
    def net_score(self) -> float:
        """Range: -1.0 (very negative) to +1.0 (very positive)."""
        net = self.positive_count - self.negative_count
        return round(net / self.total_words, 4) if self.total_words else 0.0

    @property
    def polarity(self) -> str:
        score = self.net_score
        if score >  0.005: return "positive"
        if score < -0.005: return "negative"
        return "neutral"

    @property
    def confidence(self) -> float:
        """Confidence based on sample size."""
        signals = self.positive_count + self.negative_count
        if signals < 5:   return 0.3
        if signals < 20:  return 0.6
        if signals < 50:  return 0.8
        return 0.95

    def to_dict(self) -> Dict:
        return {
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "total_words":    self.total_words,
            "net_score":      self.net_score,
            "polarity":       self.polarity,
            "confidence":     self.confidence,
        }


class EarningsTranscript:
    """A parsed earnings call transcript."""

    def __init__(
        self,
        ticker:        str,
        fiscal_period: str,
        call_date:     str,
        segments:      Optional[List[TranscriptSegment]] = None,
        raw_text:      str = "",
    ) -> None:
        self.ticker        = ticker.upper()
        self.fiscal_period = fiscal_period
        self.call_date     = call_date
        self.segments      = segments or []
        self.raw_text      = raw_text

    @property
    def full_text(self) -> str:
        if self.raw_text:
            return self.raw_text
        return "\n\n".join(
            f"{s.speaker}: {s.text}" for s in self.segments
        )

    def get_speakers(self) -> List[str]:
        seen = set()
        out  = []
        for s in self.segments:
            if s.speaker not in seen:
                seen.add(s.speaker)
                out.append(s.speaker)
        return out

    def get_ceo_segments(self) -> List[TranscriptSegment]:
        return [
            s for s in self.segments
            if "ceo" in s.role.lower() or "chief executive" in s.role.lower()
        ]

    def get_cfo_segments(self) -> List[TranscriptSegment]:
        return [
            s for s in self.segments
            if "cfo" in s.role.lower() or "chief financial" in s.role.lower()
        ]

    def to_dict(self) -> Dict:
        return {
            "ticker":        self.ticker,
            "fiscal_period": self.fiscal_period,
            "call_date":     self.call_date,
            "segment_count": len(self.segments),
            "speakers":      self.get_speakers(),
        }


class TranscriptAnalyser:
    """
    Phase 7C — Earnings call transcript parser + analyser.

    Features:
        - Parse raw HTML/TXT transcripts into speaker segments
        - Rule-based sentiment scoring (positive/negative lexicons)
        - Forward guidance extraction
        - Risk signal detection
        - Local caching
    """

    def __init__(
        self,
        cache_dir: str  = DEFAULT_CACHE_DIR,
        enabled:   bool = TRANSCRIPTS_ENABLED,
    ) -> None:
        self.cache_dir = cache_dir
        self.enabled   = enabled
        os.makedirs(cache_dir, exist_ok=True)

    # ── Parsing ───────────────────────────────────────────────────────────────

    def parse_text(
        self,
        raw_text:      str,
        ticker:        str,
        fiscal_period: str = "",
        call_date:     str = "",
    ) -> EarningsTranscript:
        """
        Parse raw transcript text into structured EarningsTranscript.
        Expected format: SPEAKER: text
        """
        segments = []
        pattern  = re.compile(
            r"^([A-Z][A-Za-z0-9\.\-,' ]{2,80}?)\s*:\s*(.+)$",
            re.MULTILINE,
        )

        current_speaker = ""
        current_text    = []
        position        = 0

        for line in raw_text.split("\n"):
            line = line.strip()
            if not line:
                continue

            m = pattern.match(line)
            if m:
                # Save previous speaker segment
                if current_speaker and current_text:
                    segments.append(TranscriptSegment(
                        speaker  = current_speaker,
                        text     = " ".join(current_text),
                        role     = self._infer_role(current_speaker),
                        position = position,
                    ))
                    position += 1
                current_speaker = m.group(1).strip()
                current_text    = [m.group(2).strip()]
            else:
                if current_speaker:
                    current_text.append(line)

        # Last segment
        if current_speaker and current_text:
            segments.append(TranscriptSegment(
                speaker  = current_speaker,
                text     = " ".join(current_text),
                role     = self._infer_role(current_speaker),
                position = position,
            ))

        return EarningsTranscript(
            ticker        = ticker,
            fiscal_period = fiscal_period,
            call_date     = call_date,
            segments      = segments,
            raw_text      = raw_text,
        )

    # ── Analysis ──────────────────────────────────────────────────────────────

    def score_sentiment(self, text: str) -> SentimentScore:
        """Rule-based sentiment scoring."""
        text_lower = text.lower()
        words      = re.findall(r"\b[a-z]+\b", text_lower)

        pos = sum(1 for w in words if w in POSITIVE_WORDS)
        neg = sum(1 for w in words if w in NEGATIVE_WORDS)

        return SentimentScore(
            positive_count = pos,
            negative_count = neg,
            total_words    = len(words),
        )

    def extract_guidance(self, text: str) -> List[str]:
        """Extract forward-guidance sentences."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        guidance  = []
        for s in sentences:
            s_lower = s.lower()
            if any(kw in s_lower for kw in GUIDANCE_KEYWORDS):
                if len(s) < 400:
                    guidance.append(s.strip())
        return guidance

    def extract_risks(self, text: str) -> List[str]:
        """Extract risk-mention sentences."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        risks     = []
        for s in sentences:
            s_lower = s.lower()
            if any(kw in s_lower for kw in RISK_KEYWORDS):
                if len(s) < 400:
                    risks.append(s.strip())
        return risks

    def analyse(self, transcript: EarningsTranscript) -> Dict:
        """Full analysis of an EarningsTranscript."""
        text      = transcript.full_text
        sentiment = self.score_sentiment(text)
        guidance  = self.extract_guidance(text)
        risks     = self.extract_risks(text)

        ceo_segs = transcript.get_ceo_segments()
        cfo_segs = transcript.get_cfo_segments()

        ceo_sentiment = self.score_sentiment(
            " ".join(s.text for s in ceo_segs)
        ) if ceo_segs else None
        cfo_sentiment = self.score_sentiment(
            " ".join(s.text for s in cfo_segs)
        ) if cfo_segs else None

        return {
            "ticker":            transcript.ticker,
            "fiscal_period":     transcript.fiscal_period,
            "overall_sentiment": sentiment.to_dict(),
            "ceo_sentiment":     ceo_sentiment.to_dict() if ceo_sentiment else None,
            "cfo_sentiment":     cfo_sentiment.to_dict() if cfo_sentiment else None,
            "guidance_count":    len(guidance),
            "guidance":          guidance[:10],
            "risk_count":        len(risks),
            "risks":             risks[:10],
            "speakers":          transcript.get_speakers(),
        }

    # ── Cache ─────────────────────────────────────────────────────────────────

    def save_transcript(self, transcript: EarningsTranscript) -> None:
        path = self._cache_path(transcript.ticker, transcript.fiscal_period)
        data = {
            "ticker":        transcript.ticker,
            "fiscal_period": transcript.fiscal_period,
            "call_date":     transcript.call_date,
            "raw_text":      transcript.raw_text,
            "segments":      [s.to_dict() for s in transcript.segments],
        }
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(data, fp)

    def load_transcript(
        self, ticker: str, fiscal_period: str
    ) -> Optional[EarningsTranscript]:
        path = self._cache_path(ticker, fiscal_period)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            segs = [
                TranscriptSegment(
                    speaker  = s["speaker"],
                    text     = s["text"],
                    role     = s.get("role",     "Unknown"),
                    position = s.get("position", 0),
                )
                for s in data.get("segments", [])
            ]
            return EarningsTranscript(
                ticker        = data["ticker"],
                fiscal_period = data.get("fiscal_period", ""),
                call_date     = data.get("call_date",     ""),
                segments      = segs,
                raw_text      = data.get("raw_text",      ""),
            )
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("[7C Transcripts] Cache read failed: %s", exc)
            return None

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _infer_role(speaker: str) -> str:
        """Best-effort role inference from speaker line."""
        s = speaker.lower()
        if "ceo"        in s or "chief executive" in s: return "CEO"
        if "cfo"        in s or "chief financial" in s: return "CFO"
        if "coo"        in s or "chief operating" in s: return "COO"
        if "analyst"    in s:                            return "Analyst"
        if "operator"   in s:                            return "Operator"
        return "Executive"

    def _cache_path(self, ticker: str, fiscal_period: str) -> str:
        safe_fp = re.sub(r"[^\w\-]", "_", fiscal_period) or "latest"
        return os.path.join(
            self.cache_dir, f"{ticker.upper()}_{safe_fp}.json"
        )