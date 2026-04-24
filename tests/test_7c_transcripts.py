"""
tests/test_7c_transcripts.py
Tests for Phase 7C Earnings Call Transcripts
PDR-BAAAI-001 Rev 1.0
20 tests - no network needed
"""
import os
import pytest
from src.live_data.transcripts import (
    TranscriptAnalyser,
    TranscriptSegment,
    SentimentScore,
    EarningsTranscript,
    POSITIVE_WORDS,
    NEGATIVE_WORDS,
    GUIDANCE_KEYWORDS,
    RISK_KEYWORDS,
    SEED,
)


SAMPLE_TRANSCRIPT = """
Operator: Welcome to Apple's Q4 2023 earnings conference call.

Tim Cook, CEO: Thank you. Today we are reporting record revenue of 383 billion
dollars, which is a strong performance. We are confident about our outlook for
fiscal 2024 and expect continued growth in services. We faced some headwinds in
China but our overall momentum remains robust.

Luca Maestri, CFO: For Q1 2024, we expect revenue to be similar to last year.
Our gross margin guidance is between 45 and 46 percent. We anticipate foreign
exchange headwinds of approximately 100 basis points.

Analyst 1: Can you comment on the iPhone demand in India?

Tim Cook, CEO: India is a great opportunity for us. We set a record there this
quarter with strong double digit growth.
"""


@pytest.fixture
def cache_dir(tmp_path):
    return str(tmp_path / "transcripts_cache")


@pytest.fixture
def analyser(cache_dir):
    return TranscriptAnalyser(cache_dir=cache_dir, enabled=False)


class TestConstants:

    def test_01_positive_words_defined(self):
        assert "strong" in POSITIVE_WORDS
        assert "record" in POSITIVE_WORDS
        assert "growth" in POSITIVE_WORDS

    def test_02_negative_words_defined(self):
        assert "weak"     in NEGATIVE_WORDS
        assert "headwinds" in NEGATIVE_WORDS
        assert "decline"  in NEGATIVE_WORDS

    def test_03_guidance_keywords_defined(self):
        assert "guidance" in GUIDANCE_KEYWORDS
        assert "outlook"  in GUIDANCE_KEYWORDS

    def test_04_risk_keywords_defined(self):
        assert "risk"        in RISK_KEYWORDS
        assert "uncertainty" in RISK_KEYWORDS

    def test_05_seed_is_42(self):
        assert SEED == 42


class TestSentimentScore:

    def test_06_net_score_positive(self):
        s = SentimentScore(positive_count=10, negative_count=2, total_words=100)
        assert s.net_score > 0

    def test_07_net_score_negative(self):
        s = SentimentScore(positive_count=2, negative_count=10, total_words=100)
        assert s.net_score < 0

    def test_08_polarity_positive(self):
        s = SentimentScore(positive_count=10, negative_count=2, total_words=100)
        assert s.polarity == "positive"

    def test_09_polarity_negative(self):
        s = SentimentScore(positive_count=2, negative_count=10, total_words=100)
        assert s.polarity == "negative"

    def test_10_to_dict_has_keys(self):
        d = SentimentScore(5, 3, 100).to_dict()
        assert "polarity"   in d
        assert "net_score"  in d
        assert "confidence" in d


class TestParsing:

    def test_11_parse_returns_transcript(self, analyser):
        t = analyser.parse_text(SAMPLE_TRANSCRIPT, "AAPL", "FY2023_Q4")
        assert isinstance(t, EarningsTranscript)

    def test_12_parse_finds_segments(self, analyser):
        t = analyser.parse_text(SAMPLE_TRANSCRIPT, "AAPL")
        assert len(t.segments) >= 4

    def test_13_parse_identifies_ceo(self, analyser):
        t   = analyser.parse_text(SAMPLE_TRANSCRIPT, "AAPL")
        ceo = t.get_ceo_segments()
        assert len(ceo) >= 1

    def test_14_parse_identifies_cfo(self, analyser):
        t   = analyser.parse_text(SAMPLE_TRANSCRIPT, "AAPL")
        cfo = t.get_cfo_segments()
        assert len(cfo) >= 1


class TestAnalysis:

    def test_15_score_sentiment_detects_positive(self, analyser):
        text = "record strong growth momentum bullish confident"
        s    = analyser.score_sentiment(text)
        assert s.polarity == "positive"

    def test_16_score_sentiment_detects_negative(self, analyser):
        text = "weak decline headwinds challenging missed below disappointing"
        s    = analyser.score_sentiment(text)
        assert s.polarity == "negative"

    def test_17_extract_guidance_finds_sentences(self, analyser):
        g = analyser.extract_guidance(SAMPLE_TRANSCRIPT)
        assert len(g) >= 1

    def test_18_extract_risks_finds_sentences(self, analyser):
        r = analyser.extract_risks(SAMPLE_TRANSCRIPT)
        assert len(r) >= 1

    def test_19_analyse_full_transcript(self, analyser):
        t      = analyser.parse_text(SAMPLE_TRANSCRIPT, "AAPL", "FY2023_Q4")
        result = analyser.analyse(t)
        assert "overall_sentiment" in result
        assert "guidance"          in result
        assert "risks"             in result
        assert result["ticker"] == "AAPL"


class TestCache:

    def test_20_save_and_load(self, analyser):
        original = analyser.parse_text(SAMPLE_TRANSCRIPT, "AAPL", "FY2023_Q4")
        analyser.save_transcript(original)
        loaded   = analyser.load_transcript("AAPL", "FY2023_Q4")
        assert loaded is not None
        assert loaded.ticker == "AAPL"
        assert len(loaded.segments) == len(original.segments)