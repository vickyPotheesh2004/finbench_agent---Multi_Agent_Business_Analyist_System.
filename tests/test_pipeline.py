"""
tests/test_pipeline.py already exists — skip if present.
This is a smoke test only — full integration test in test_integration.py
"""
import pytest
from src.pipeline.pipeline import FinBenchPipeline, run_pipeline


class TestPipelineImport:

    def test_01_pipeline_imports(self):
        from src.pipeline.pipeline import FinBenchPipeline
        assert FinBenchPipeline is not None

    def test_02_run_pipeline_imports(self):
        from src.pipeline.pipeline import run_pipeline
        assert run_pipeline is not None

    def test_03_pipeline_instantiates(self):
        p = FinBenchPipeline()
        assert p is not None
        assert p._built is False

    def test_04_pipeline_not_built_until_needed(self):
        p = FinBenchPipeline()
        assert p._ingestion_graph is None
        assert p._query_graph is None