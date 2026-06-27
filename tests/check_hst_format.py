"""Quick check: what SQLite journal mode does pyOptsparse use for .hst files?"""
import sqlite3, tempfile, os, sys
sys.path.insert(0, r"C:\Users\benji\dymos-rtplot\src")

try:
    from pyoptsparse.pyOpt_history import History
except ImportError:
    print("pyoptsparse not installed")
    sys.exit(1)

with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
    hst = os.path.join(td, "test.hst")
    h = History(hst, flag="n")
    # pyoptsparse 2.x API: writeData writes metadata, write writes a full entry.
    # We just need the file to exist; close() flushes.
    h.close()

    # Read SQLite metadata directly
    try:
        con = sqlite3.connect(hst)
        mode    = con.execute("PRAGMA journal_mode").fetchone()[0]
        page_sz = con.execute("PRAGMA page_size").fetchone()[0]
        tables  = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        con.close()
        print(f"journal_mode : {mode}")
        print(f"page_size    : {page_sz}")
        print(f"tables       : {tables}")
    except sqlite3.DatabaseError as e:
        print(f"Not a SQLite file (or can't open): {e}")
        # Read first 16 bytes to identify file format
        with open(hst, "rb") as f:
            header = f.read(16)
        print(f"File header  : {header!r}")
