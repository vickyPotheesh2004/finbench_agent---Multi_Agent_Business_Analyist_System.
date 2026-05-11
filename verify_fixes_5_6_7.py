"""Verify all three quick fixes are applied."""
from pathlib import Path

checks = {}

# Fix 5 — BM25 noise
bm = Path('src/retrieval/bm25_retriever.py').read_text(encoding='utf-8')
checks['Fix 5: Bug Fix 5 marker']     = 'Bug Fix 5' in bm
checks['Fix 5: print removed']        = 'print(f"[N07] BM25 retrieve failed' not in bm
checks['Fix 5: logger.debug present'] = 'logger.debug("[N07] bm25s retrieve' in bm or 'logging.getLogger(__name__).debug' in bm

# Fix 6 — x10^6 -> million
sn = Path('src/retrieval/sniper_rag.py').read_text(encoding='utf-8')
checks['Fix 6: _UNIT_DISPLAY map']    = '_UNIT_DISPLAY' in sn
checks['Fix 6: million mapping']      = '"million"' in sn
checks['Fix 6: Bug Fix 6 marker']     = 'Bug Fix 6' in sn

# Fix 7 — SHAP noise
sh = Path('src/analysis/shap_dag.py').read_text(encoding='utf-8')
checks['Fix 7: Bug Fix 7 marker']     = 'Bug Fix 7' in sh
checks['Fix 7: filter tiny texts']    = 'len(t.strip()) > 30' in sh
checks['Fix 7: demoted to debug']     = 'SHAP computation skipped' in sh

# Print results
print()
print("=" * 60)
print(" VERIFY FIXES 5, 6, 7")
print("=" * 60)
for name, passed in checks.items():
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}")
print("-" * 60)
total = len(checks)
passed_count = sum(1 for v in checks.values() if v)
print(f"  Total: {passed_count}/{total} passing")
print()

if passed_count == total:
    print("All 3 quick fixes applied successfully.")
else:
    print(f"  {total - passed_count} check(s) failed — review above.")