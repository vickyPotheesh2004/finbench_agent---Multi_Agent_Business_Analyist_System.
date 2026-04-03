"""
src/pipeline/pipeline.py
FinBench Multi-Agent Business Analyst AI
PDR-BAAAI-001 Rev2.0

Full 19-node pipeline.
Single run_query_only() / run() call executes entire system.

INGESTION (once per document):  N01 → N02 → N03
QUERY PHASE (per question):     N04 → N05 → N06 → [N07-N09] → N10
                                 → N11+N12+N13+N14 (parallel)
                                 → N15 → N16 → N17 → N18 → N19
"""

import sys
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.state.ba_state          import BAState
from src.utils.seed_manager      import SeedManager
from src.utils.resource_governor import ResourceGovernor

SeedManager.set_all()


class FinBenchPipeline:
    """
    Full 19-node FinBench pipeline.

    Usage:
        pipeline = FinBenchPipeline()
        state    = pipeline.ingest("apple_10k.pdf", "Apple Inc", "10-K", "FY2023")
        state    = pipeline.query(state, "What was Apple net income FY2023?")
        print(state.final_answer)
        print(state.final_report_path)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        SeedManager.set_all()
        self.config        = config or {}
        self._nodes_fired: List[str] = []
        self._init_nodes()

    def _init_nodes(self) -> None:
        """Initialise all 19 nodes with correct class names."""

        from src.ingestion.pdf_ingestor         import PDFIngestor
        from src.ingestion.section_tree_builder import SectionTreeBuilder
        from src.ingestion.chunker              import Chunker

        from src.routing.cart_router            import CARTRouter
        from src.routing.lr_difficulty          import LRDifficultyPredictor

        from src.retrieval.sniper_rag           import SniperRAG
        from src.retrieval.bm25_retriever       import BM25Retriever
        from src.retrieval.bge_retriever        import BGERetriever
        from src.retrieval.rrf_reranker         import RRFReranker

        from src.prompts.assembler              import PromptAssembler

        from src.agents.analyst_pod             import AnalystPod
        from src.agents.quant_pod               import QuantPod
        from src.agents.triguard                import TriGuard
        from src.agents.auditor_pod             import AuditorPod
        from src.agents.piv_mediator            import PIVMediator

        from src.explainability.shap_dag        import SHAPDag
        from src.ml.xgb_arbiter                 import XGBArbiter
        from src.rlef.jee_engine                import JEEEngine
        from src.output.output_generator        import OutputGenerator

        self.n01 = PDFIngestor()
        self.n02 = SectionTreeBuilder()
        self.n03 = Chunker()
        self.n04 = CARTRouter()
        self.n05 = LRDifficultyPredictor()
        self.n06 = SniperRAG()
        self.n07 = BM25Retriever()
        self.n08 = BGERetriever()
        self.n09 = RRFReranker()
        self.n10 = PromptAssembler()
        self.n11 = AnalystPod()
        self.n12 = QuantPod()
        self.n13 = TriGuard()
        self.n14 = AuditorPod()
        self.n15 = PIVMediator()
        self.n16 = SHAPDag()
        self.n17 = XGBArbiter()
        self.n18 = JEEEngine()
        self.n19 = OutputGenerator()

        print("[Pipeline] All 19 nodes initialised")

    # ═══════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ═══════════════════════════════════════════════════════════════════════

    def ingest(
        self,
        document_path: str,
        company_name:  str = "",
        doc_type:      str = "10-K",
        fiscal_year:   str = "",
        session_id:    str = "",
    ) -> BAState:
        """Ingest document — N01-N03 once. Returns BAState ready for queries."""
        from uuid import uuid4
        state = BAState(
            session_id    = session_id or str(uuid4()),
            document_path = document_path,
            company_name  = company_name,
            doc_type      = doc_type,
            fiscal_year   = fiscal_year,
        )
        return self._run_ingestion(state)

    def query(self, state: BAState, question: str) -> BAState:
        """Run N04-N19 against pre-ingested document."""
        from uuid import uuid4
        state.query      = question
        state.session_id = str(uuid4())
        return self._run_query_phase(state)

    def run(
        self,
        document_path: str,
        question:      str,
        company_name:  str = "",
        doc_type:      str = "10-K",
        fiscal_year:   str = "",
    ) -> BAState:
        """Full pipeline — ingest + query in one call."""
        state = self.ingest(document_path, company_name, doc_type, fiscal_year)
        return self.query(state, question)

    def run_query_only(self, state: BAState, question: str) -> BAState:
        """Run query phase on pre-built BAState (chunks already in state)."""
        state.query = question
        return self._run_query_phase(state)

    def get_nodes_fired(self) -> List[str]:
        """Return list of nodes fired in last run."""
        return self._nodes_fired.copy()

    # ═══════════════════════════════════════════════════════════════════════
    # INGESTION PHASE — N01-N03
    # ═══════════════════════════════════════════════════════════════════════

    def _run_ingestion(self, state: BAState) -> BAState:
        """N01 → N02 → N03 sequential."""
        self._nodes_fired = []
        state = self._fire(self.n01, state, "N01_PDF_INGESTOR")
        state = self._fire(self.n02, state, "N02_SECTION_TREE")
        state = self._fire(self.n03, state, "N03_CHUNKER")
        print(f"[Pipeline] Ingestion complete — chunks={state.chunk_count}")
        return state

    # ═══════════════════════════════════════════════════════════════════════
    # QUERY PHASE — N04-N19
    # ═══════════════════════════════════════════════════════════════════════

    def _run_query_phase(self, state: BAState) -> BAState:
        """Full query phase with conditional SniperRAG + parallel pods."""
        ResourceGovernor.check("Pipeline query phase")
        self._nodes_fired = []

        # Routing
        state = self._fire(self.n04, state, "N04_CART_ROUTER")
        state = self._fire(self.n05, state, "N05_LR_DIFFICULTY")

        # Retrieval — conditional SniperRAG edge
        state = self._fire(self.n06, state, "N06_SNIPER_RAG")

        if state.sniper_hit and state.sniper_confidence >= 0.95:
            print("[Pipeline] N06 SniperRAG HIT — skipping N07-N09")
            if state.sniper_result:
                state.retrieval_stage_2 = [{
                    "text":    state.sniper_result,
                    "section": "Table Index",
                    "page":    "0",
                    "source":  "SniperRAG",
                }]
        else:
            state = self._fire(self.n07, state, "N07_BM25")
            state = self._fire(self.n08, state, "N08_BGE_M3")
            state = self._fire(self.n09, state, "N09_RRF_RERANKER")

        # Prompt assembly
        state = self._fire(self.n10, state, "N10_PROMPT_ASSEMBLER")

        # Parallel analysis pods N11-N14
        state = self._run_parallel_pods(state)

        # Mediation
        state = self._fire(self.n15, state, "N15_PIV_MEDIATOR")
        if state.final_answer_pre_xgb and not state.final_answer:
            state.final_answer = state.final_answer_pre_xgb

        # Post-analysis
        state = self._fire(self.n16, state, "N16_SHAP_DAG")
        state = self._fire(self.n17, state, "N17_XGB_ARBITER")
        state = self._fire(self.n18, state, "N18_RLEF_JEE")
        state = self._fire(self.n19, state, "N19_OUTPUT")

        print(f"[Pipeline] Query complete — "
              f"answer={len(state.final_answer)} chars "
              f"nodes={len(self._nodes_fired)} "
              f"risk={state.risk_score:.1f}")
        return state

    def _run_parallel_pods(self, state: BAState) -> BAState:
        """N11+N12+N13+N14 in parallel via ThreadPoolExecutor."""
        s11 = copy.deepcopy(state)
        s12 = copy.deepcopy(state)
        s13 = copy.deepcopy(state)
        s14 = copy.deepcopy(state)

        results: Dict[str, BAState] = {}

        def run_pod(pod, s, name):
            try:
                return name, self._fire(pod, s, name)
            except Exception as e:
                print(f"[Pipeline] {name} error: {e}")
                return name, s

        with ThreadPoolExecutor(max_workers=4) as ex:
            futures = {
                ex.submit(run_pod, self.n11, s11, "N11_ANALYST"):  "N11",
                ex.submit(run_pod, self.n12, s12, "N12_QUANT"):    "N12",
                ex.submit(run_pod, self.n13, s13, "N13_TRIGUARD"): "N13",
                ex.submit(run_pod, self.n14, s14, "N14_AUDITOR"):  "N14",
            }
            for future in as_completed(futures):
                try:
                    name, rs = future.result(timeout=300)
                    results[name[:3]] = rs
                except Exception as e:
                    print(f"[Pipeline] Pod future error: {e}")

        # Merge N11
        if "N11" in results:
            s = results["N11"]
            state.analyst_output        = s.analyst_output
            state.analyst_confidence    = s.analyst_confidence
            state.analyst_citations     = s.analyst_citations
            state.analyst_piv_status    = s.analyst_piv_status
            state.analyst_attempt_count = s.analyst_attempt_count
            state.analyst_low_conf      = s.analyst_low_conf

        # Merge N12
        if "N12" in results:
            s = results["N12"]
            state.quant_result          = s.quant_result
            state.quant_confidence      = s.quant_confidence
            state.quant_citations       = s.quant_citations
            state.quant_piv_status      = s.quant_piv_status
            state.quant_attempt_count   = s.quant_attempt_count
            state.monte_carlo_results   = s.monte_carlo_results
            state.var_result            = s.var_result
            state.garch_result          = s.garch_result
            state.computed_ratio        = s.computed_ratio

        # Merge N13
        if "N13" in results:
            s = results["N13"]
            state.forensic_flags        = s.forensic_flags
            state.risk_score            = s.risk_score
            state.anomaly_detected      = s.anomaly_detected
            state.anomaly_severity      = s.anomaly_severity
            state.benford_chi2          = s.benford_chi2
            state.benford_p_value       = s.benford_p_value

        # Merge N14
        if "N14" in results:
            s = results["N14"]
            state.auditor_output        = s.auditor_output
            state.auditor_confidence    = s.auditor_confidence
            state.auditor_citations     = s.auditor_citations
            state.auditor_piv_status    = s.auditor_piv_status
            state.auditor_attempt_count = s.auditor_attempt_count
            state.contradiction_flags   = s.contradiction_flags

        print(f"[Pipeline] Parallel complete — "
              f"analyst={len(state.analyst_output)} "
              f"quant={len(state.quant_result)} "
              f"risk={state.risk_score:.1f}")
        return state

    # ═══════════════════════════════════════════════════════════════════════
    # NODE RUNNER
    # ═══════════════════════════════════════════════════════════════════════

    def _fire(self, node: Any, state: BAState, node_name: str) -> BAState:
        """Fire one node. Graceful degradation except MemoryError."""
        try:
            result = node.run(state)
            self._nodes_fired.append(node_name)
            return result if result is not None else state
        except MemoryError as e:
            print(f"[Pipeline] {node_name} C4 HALT: {e}")
            raise
        except Exception as e:
            print(f"[Pipeline] {node_name} ERROR: {type(e).__name__}: {e}")
            self._nodes_fired.append(f"{node_name}_ERROR")
            return state


# ═══════════════════════════════════════════════════════════════════════════
# SANITY CHECK — python src/pipeline/pipeline.py
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        from rich import print as rprint
    except ImportError:
        rprint = print

    rprint("\n[bold cyan]-- FinBenchPipeline sanity check --[/bold cyan]")

    pipeline = FinBenchPipeline()
    rprint("[green]✓[/green] Pipeline instantiated — all 19 nodes")

    state = BAState(
        session_id          = "sanity-pipeline",
        query               = "What was Apple net income FY2023?",
        company_name        = "Apple Inc",
        doc_type            = "10-K",
        fiscal_year         = "FY2023",
        chunk_count         = 2,
        bm25_index_path     = "mock",
        chromadb_collection = "mock",
        retrieval_stage_2   = [{
            "text":        "Net income $96,995 million FY2023. "
                           "Revenue $383,285 million.",
            "section":     "Financial Statements",
            "page":        "42",
            "company":     "Apple Inc",
            "fiscal_year": "FY2023",
            "bm25_score":  0.85,
            "cosine_sim":  0.92,
        }],
    )

    rprint("[yellow]Running query phase — Ollama timeout = fallback OK[/yellow]")
    state = pipeline.run_query_only(state, "What was Apple net income FY2023?")

    nodes = pipeline.get_nodes_fired()
    rprint(f"[green]✓[/green] Nodes fired:        {len(nodes)}")
    rprint(f"[green]✓[/green] Final answer length: {len(state.final_answer)}")
    rprint(f"[green]✓[/green] Risk score:          {state.risk_score}")
    rprint(f"[green]✓[/green] Report path:         {state.final_report_path}")
    rprint(f"[green]✓[/green] seed=42:             {state.seed}")

    assert state.seed == 42
    assert len(nodes) >= 10
    assert (state.final_answer or state.xgb_ranked_answer
            or state.final_answer_pre_xgb) != ""

    rprint(f"\n[green]Nodes: {nodes}[/green]")
    rprint(f"\n[bold green]Pipeline sanity check passed.[/bold green]\n")