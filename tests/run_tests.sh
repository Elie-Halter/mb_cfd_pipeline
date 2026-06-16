#!/usr/bin/env bash
# Run the MB-CFD test suite. No patient data and no solver required:
# every tool is exercised on a tiny synthetic case, plus compile/import smoke tests.
#   bash tests/run_tests.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."

FILT='WARN|PyVistaFutureWarning|will change|silence|^  b = |^  sub = |degenerate|^$'
fail=0
for t in tests/test_imports.py tests/test_postproc.py tests/test_meshing.py; do
    echo "### $t"
    python3 "$t" 2>&1 | grep -vE "$FILT"
    [ "${PIPESTATUS[0]}" -ne 0 ] && fail=1
    echo
done

if [ "$fail" -eq 0 ]; then
    echo "============================================================"
    echo "  ALL TESTS PASSED"
    echo "============================================================"
else
    echo "============================================================"
    echo "  SOME TESTS FAILED"
    echo "============================================================"
fi
exit $fail
