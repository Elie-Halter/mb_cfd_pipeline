"""
Smoke test: every Python file in morph/ and tools/ compiles, and the key library
modules import cleanly (catches syntax errors, broken imports, missing deps).
Run:  python3 tests/test_imports.py
"""
import sys, os, py_compile, importlib, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    pyfiles = sorted(glob.glob(os.path.join(ROOT, "morph", "*.py")) +
                     glob.glob(os.path.join(ROOT, "tools", "*.py")))
    nfail = 0
    print(f"compiling {len(pyfiles)} files ...")
    for f in pyfiles:
        try:
            py_compile.compile(f, doraise=True)
        except py_compile.PyCompileError as e:
            print(f"  [FAIL] compile {os.path.relpath(f, ROOT)}: {e}"); nfail += 1

    # import the key library modules (must load without a __main__ run)
    for sub in ("morph", "tools"):
        sys.path.insert(0, os.path.join(ROOT, sub))
    for mod in ["register", "morph_volume", "check_write_any",
                "hemo_indices", "compare_FB_MB", "gci", "build_iso_mesh"]:
        try:
            importlib.import_module(mod)
            print(f"  [PASS] import {mod}")
        except Exception as e:
            print(f"  [FAIL] import {mod}: {e}"); nfail += 1

    print(f"\n=== imports/compile: {'ALL OK' if nfail == 0 else str(nfail) + ' FAIL'} ===")
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
