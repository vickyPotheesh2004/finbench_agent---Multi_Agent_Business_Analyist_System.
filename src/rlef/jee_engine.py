"""
N18 RLEF JEE Engine — Joint Evaluation Engine
PDR-BAAAI-001 · Rev 1.0 · Node N18

Purpose:
    Grades every session with 3 validators:
        V-A: Numerical Precision   (+4 / +2 / -1)
        V-B: Citation Quality      (+4 / +2 / -1)
        V-C: Answer Completeness   (+4 / +2 / -1)
    Stores DPO pairs (chosen/rejected) in local SQLite.
    All _rlef_ fields are PRIVATE — never in any output (C9).

Grading scale:
    +4  = fully correct
    +2  = partially correct
    -1  = wrong / hallucinated

Constraints satisfied:
    C1  $0 cost — sqlite3 is Python stdlib
    C2  100% local — zero network calls
    C5  seed=42
    C9  _rlef_ fields NEVER in output
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

SEED          = 42
DB_PATH       = "data/rlef_training.db"
GRADE_CORRECT = 4
GRADE_PARTIAL = 2
GRADE_WRONG   = -1

# Minimum grade to keep a DPO pair (reject -1 graded pairs)
MIN_GRADE_FOR_DPO = 0

# Completeness thresholds
COMPLETENESS_CORRECT_CHARS = 75   # >= 75 chars → CORRECT
COMPLETENESS_PARTIAL_CHARS = 30   # >= 30 chars → PARTIAL


@dataclass
class RLEFGrade:
    va_score:      int
    vb_score:      int
    vc_score:      int
    total_grade:   int
    grade_label:   str    # 'correct' / 'partial' / 'wrong'
    session_id:    str
    timestamp_utc: str


@dataclass
class DPOPair:
    session_id:  str
    query:       str
    chosen:      str
    rejected:    str
    grade_delta: int
    query_type:  str
    difficulty:  str


class RLEFJEEEngine:
    """
    N18 RLEF JEE Engine.
    Grades sessions, stores DPO pairs, feeds the self-improvement loop.
    """

    def __init__(self, db_path: str = DB_PATH) -> None:
        self.db_path = db_path
        self._ensure_db()

    # ── LangGraph pipeline node ───────────────────────────────────────────────

    def run(self, state) -> object:
        """
        LangGraph N18 node entry point.
        Reads:  state.final_answer_pre_xgb, state.query,
                state.analyst_citations, state.confidence_score
        Writes: state._rlef_grade, state._rlef_va_score,
                state._rlef_vb_score, state._rlef_vc_score  (ALL PRIVATE C9)
        """
        session_id = getattr(state, "session_id",           "") or ""
        query      = getattr(state, "query",                "") or ""
        answer     = getattr(state, "final_answer_pre_xgb", "") or ""
        citations  = getattr(state, "analyst_citations",    []) or []
        confidence = getattr(state, "confidence_score",     0.0)
        query_type = getattr(state, "query_type",           "text")
        difficulty = getattr(state, "query_difficulty",     "medium")

        if hasattr(query_type, "value"):
            query_type = query_type.value
        if hasattr(difficulty, "value"):
            difficulty = difficulty.value

        if not answer:
            logger.warning("N18: empty answer — skipping grading")
            return state

        grade = self.grade(
            session_id = session_id,
            query      = query,
            answer     = answer,
            citations  = citations,
            confidence = confidence,
            query_type = query_type,
            difficulty = difficulty,
        )

        # Write ONLY to private _rlef_ fields — C9
        state._rlef_grade    = grade.total_grade
        state._rlef_va_score = float(grade.va_score)
        state._rlef_vb_score = float(grade.vb_score)
        state._rlef_vc_score = float(grade.vc_score)

        logger.info(
            "N18 RLEF: grade=%d (%s) | VA=%d VB=%d VC=%d | session=%s",
            grade.total_grade, grade.grade_label,
            grade.va_score, grade.vb_score, grade.vc_score,
            session_id[:8],
        )
        return state

    # ── Core grading method ───────────────────────────────────────────────────

    def grade(
        self,
        session_id:  str,
        query:       str,
        answer:      str,
        citations:   List[str] = None,
        confidence:  float     = 0.0,
        gold:        str       = "",
        query_type:  str       = "text",
        difficulty:  str       = "medium",
    ) -> RLEFGrade:
        """
        Grade an answer using 3 validators.
        Stores result in SQLite automatically.
        """
        citations = citations or []

        va = self._grade_numerical_precision(answer, query, gold)
        vb = self._grade_citation_quality(answer, citations)
        vc = self._grade_completeness(answer, query)

        total = va + vb + vc

        if total >= 8:
            label = "correct"
        elif total >= 3:
            label = "partial"
        else:
            label = "wrong"

        grade = RLEFGrade(
            va_score      = va,
            vb_score      = vb,
            vc_score      = vc,
            total_grade   = total,
            grade_label   = label,
            session_id    = session_id,
            timestamp_utc = datetime.utcnow().isoformat(),
        )

        # Always store — grade() is the single entry point for storage
        self._store_grade(grade, query, answer, query_type, difficulty)

        return grade

    # ── DPO extraction ────────────────────────────────────────────────────────

    def extract_dpo_pairs(
        self,
        min_grade: int = MIN_GRADE_FOR_DPO,
        limit:     int = 500,
    ) -> List[DPOPair]:
        """
        Extract DPO (chosen/rejected) pairs from SQLite for training.
        Pairs sessions with same query_type where grades differ.
        Only returns pairs where grade >= min_grade.
        """
        conn   = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT session_id, query, answer, total_grade, query_type, difficulty
            FROM rlef_sessions
            WHERE total_grade >= ?
            ORDER BY total_grade DESC, timestamp_utc DESC
            LIMIT ?
        """, (min_grade, limit * 2))
        rows  = cursor.fetchall()
        conn.close()

        by_type: Dict[str, list] = {}
        for row in rows:
            qt = row[4] or "text"
            by_type.setdefault(qt, []).append(row)

        pairs = []
        for qt, sessions in by_type.items():
            sessions.sort(key=lambda x: x[3], reverse=True)
            i, j = 0, len(sessions) - 1
            while i < j and len(pairs) < limit:
                chosen_row   = sessions[i]
                rejected_row = sessions[j]
                if chosen_row[3] > rejected_row[3]:
                    pairs.append(DPOPair(
                        session_id  = chosen_row[0],
                        query       = chosen_row[1],
                        chosen      = chosen_row[2],
                        rejected    = rejected_row[2],
                        grade_delta = chosen_row[3] - rejected_row[3],
                        query_type  = qt,
                        difficulty  = chosen_row[5] or "medium",
                    ))
                i += 1
                j -= 1

        return pairs[:limit]

    def get_session_count(self) -> int:
        """Return total number of graded sessions in DB."""
        conn   = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM rlef_sessions")
        count  = cursor.fetchone()[0]
        conn.close()
        return count

    def get_grade_distribution(self) -> Dict[str, int]:
        """Return count of correct / partial / wrong grades."""
        conn   = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT grade_label, COUNT(*)
            FROM rlef_sessions
            GROUP BY grade_label
        """)
        rows  = cursor.fetchall()
        conn.close()
        return {row[0]: row[1] for row in rows}

    # ── Validators ────────────────────────────────────────────────────────────

    @staticmethod
    def _grade_numerical_precision(
        answer: str, query: str, gold: str = ""
    ) -> int:
        """
        V-A: Grade numerical precision.
        +4 = numbers present with units (or gold match)
        +2 = numbers present but no units
        -1 = no numbers when question requires them
        """
        has_numbers = bool(re.search(r'\$?[\d,]+(?:\.\d+)?', answer))
        has_units   = bool(re.search(
            r'million|billion|thousand|percent|%|\$', answer, re.IGNORECASE
        ))

        if gold:
            gold_nums  = re.findall(r'[\d,]+(?:\.\d+)?', gold)
            ans_nums   = re.findall(r'[\d,]+(?:\.\d+)?', answer)
            gold_clean = [n.replace(",", "") for n in gold_nums]
            ans_clean  = [n.replace(",", "") for n in ans_nums]
            if any(g in ans_clean for g in gold_clean):
                return GRADE_CORRECT

        numerical_keywords = [
            "how much", "what was", "total", "revenue", "income",
            "profit", "eps", "earnings", "sales", "cost", "expense",
        ]
        is_numerical = any(kw in query.lower() for kw in numerical_keywords)

        if is_numerical:
            if has_numbers and has_units:
                return GRADE_CORRECT
            elif has_numbers:
                return GRADE_PARTIAL
            else:
                return GRADE_WRONG
        else:
            return GRADE_CORRECT if len(answer) > 50 else GRADE_PARTIAL

    @staticmethod
    def _grade_citation_quality(
        answer: str, citations: List[str]
    ) -> int:
        """
        V-B: Grade citation quality.
        +4 = citations with section + page format
        +2 = citations present but incomplete
        -1 = no citations at all
        """
        inline_citations  = re.findall(r'\[([A-Z_]+/P\d+[^\]]*)\]', answer)
        has_section_page  = bool(inline_citations)
        has_citation_list = len(citations) > 0
        has_any_bracket   = bool(re.search(r'\[.*?\]', answer))

        if has_section_page or (has_citation_list and has_any_bracket):
            return GRADE_CORRECT
        elif has_citation_list or has_any_bracket:
            return GRADE_PARTIAL
        else:
            return GRADE_WRONG

    @staticmethod
    def _grade_completeness(answer: str, query: str) -> int:
        """
        V-C: Grade answer completeness.
        +4 = substantive answer (>= 75 chars)
        +2 = present but brief (>= 30 chars)
        -1 = empty, very short, or RETRIEVAL_MISS
        """
        if not answer or len(answer.strip()) < 10:
            return GRADE_WRONG

        if "RETRIEVAL_MISS" in answer:
            return GRADE_WRONG

        length = len(answer.strip())
        if length >= COMPLETENESS_CORRECT_CHARS:
            return GRADE_CORRECT
        elif length >= COMPLETENESS_PARTIAL_CHARS:
            return GRADE_PARTIAL
        else:
            return GRADE_WRONG

    # ── SQLite helpers ────────────────────────────────────────────────────────

    def _ensure_db(self) -> None:
        """Create SQLite DB and tables if they don't exist."""
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS rlef_sessions (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id    TEXT    NOT NULL UNIQUE,
                query         TEXT    NOT NULL,
                answer        TEXT    NOT NULL,
                va_score      INTEGER NOT NULL,
                vb_score      INTEGER NOT NULL,
                vc_score      INTEGER NOT NULL,
                total_grade   INTEGER NOT NULL,
                grade_label   TEXT    NOT NULL,
                query_type    TEXT    DEFAULT 'text',
                difficulty    TEXT    DEFAULT 'medium',
                timestamp_utc TEXT    NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_id
            ON rlef_sessions(session_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_grade
            ON rlef_sessions(total_grade)
        """)
        conn.commit()
        conn.close()

    def _store_grade(
        self,
        grade:      RLEFGrade,
        query:      str,
        answer:     str,
        query_type: str = "text",
        difficulty: str = "medium",
    ) -> None:
        """Store grade in SQLite. UNIQUE constraint on session_id — overwrites."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO rlef_sessions
            (session_id, query, answer, va_score, vb_score, vc_score,
             total_grade, grade_label, query_type, difficulty, timestamp_utc)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            grade.session_id,
            query[:500],
            answer[:2000],
            grade.va_score,
            grade.vb_score,
            grade.vc_score,
            grade.total_grade,
            grade.grade_label,
            query_type,
            difficulty,
            grade.timestamp_utc,
        ))
        conn.commit()
        conn.close()


# ── Convenience wrapper ───────────────────────────────────────────────────────

def run_rlef_engine(state, db_path: str = DB_PATH) -> object:
    """Convenience wrapper for LangGraph N18 node."""
    return RLEFJEEEngine(db_path=db_path).run(state)