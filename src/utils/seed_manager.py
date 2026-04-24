"""
src/utils/seed_manager.py
FinBench Multi-Agent Business Analyst AI
PDR-BAAAI-001 Rev1.0 FINAL

C5: seed=42 everywhere.
SeedManager wraps ALL random calls in the entire project.
Every node that needs randomness imports and uses SeedManager.
Never call random.seed() or np.random.seed() directly — always use this.
"""

import random
from typing import Optional

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ── C5: The one true seed ────────────────────────────────────────────────────
GLOBAL_SEED = 42


class SeedManager:
    """
    C5: Enforces seed=42 across all frameworks.
    Call SeedManager.set_all() at the start of every node
    that uses any randomness.
    """

    _seed: int = GLOBAL_SEED

    @classmethod
    def set_all(cls, seed: Optional[int] = None) -> int:
        """
        Set seed=42 across all frameworks.
        If seed is provided and not 42 — raises ValueError (C5).
        Returns the seed that was set (always 42).
        """
        if seed is not None and seed != GLOBAL_SEED:
            raise ValueError(
                f"[C5 VIOLATION] seed must be 42, got {seed}. "
                "Never use a seed other than 42 in this project."
            )

        cls._seed = GLOBAL_SEED

        # Python stdlib
        random.seed(GLOBAL_SEED)

        # NumPy
        np.random.seed(GLOBAL_SEED)

        # PyTorch (if installed)
        if TORCH_AVAILABLE:
            torch.manual_seed(GLOBAL_SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(GLOBAL_SEED)

        return GLOBAL_SEED

    @classmethod
    def get(cls) -> int:
        """Return the global seed (always 42)."""
        return GLOBAL_SEED

    @classmethod
    def sklearn_params(cls) -> dict:
        """
        Return sklearn-compatible seed params.
        Usage: DecisionTreeClassifier(**SeedManager.sklearn_params())
        """
        return {"random_state": GLOBAL_SEED}

    @classmethod
    def xgboost_params(cls) -> dict:
        """
        Return XGBoost-compatible seed params.
        Usage: XGBClassifier(**SeedManager.xgboost_params())
        """
        return {"seed": GLOBAL_SEED, "random_state": GLOBAL_SEED}

    @classmethod
    def numpy_rng(cls) -> np.random.Generator:
        """
        Return a seeded NumPy random Generator.
        Usage: rng = SeedManager.numpy_rng()
               rng.integers(0, 100)
        """
        return np.random.default_rng(GLOBAL_SEED)


# ═══════════════════════════════════════════════════════════════════════════
# QUICK SANITY CHECK
# run: python src/utils/seed_manager.py
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        from rich import print as rprint
    except ImportError:
        rprint = print

    rprint("\n[bold cyan]── SeedManager sanity check ──[/bold cyan]")

    # Set all seeds
    seed = SeedManager.set_all()
    rprint(f"[green]✓[/green] set_all() returned seed={seed}")

    # Get seed
    assert SeedManager.get() == 42
    rprint(f"[green]✓[/green] get() returns 42")

    # Wrong seed rejected
    try:
        SeedManager.set_all(99)
    except ValueError as e:
        rprint(f"[green]✓[/green] C5 enforced: wrong seed rejected")

    # sklearn params
    params = SeedManager.sklearn_params()
    assert params["random_state"] == 42
    rprint(f"[green]✓[/green] sklearn_params: {params}")

    # xgboost params
    params = SeedManager.xgboost_params()
    assert params["seed"] == 42
    rprint(f"[green]✓[/green] xgboost_params: {params}")

    # numpy rng
    rng = SeedManager.numpy_rng()
    val1 = rng.integers(0, 1000)
    rng2 = SeedManager.numpy_rng()
    val2 = rng2.integers(0, 1000)
    assert val1 == val2, "Same seed must produce same value"
    rprint(f"[green]✓[/green] numpy_rng reproducible: {val1} == {val2}")

    # Reproducibility check
    SeedManager.set_all()
    a = random.randint(0, 10000)
    SeedManager.set_all()
    b = random.randint(0, 10000)
    assert a == b, f"Not reproducible: {a} != {b}"
    rprint(f"[green]✓[/green] Reproducibility confirmed: {a} == {b}")

    rprint("\n[bold green]All checks passed. SeedManager ready.[/bold green]\n")