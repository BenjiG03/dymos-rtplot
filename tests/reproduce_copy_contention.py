"""
Standalone reproducer for the shutil.copy2 + SQLite malformed-disk-image race.

pyOptSparse stores its optimizer history in a SQLite database (*.hst).
The dashboard's OptimizerHistoryAccess._read_snapshot_copy() calls shutil.copy2()
on the live .hst file and then opens the copy with History().

When IPOPT is running line searches it writes a new row to the .hst file for
EVERY function evaluation.  If shutil.copy2() runs while a commit is in
progress, the copy may miss the journal (or capture a mid-write page), making
the copy "malformed" when opened.

This test reproduces that race WITHOUT needing pyoptsparse installed:
  - Thread A (writer): simulates IPOPT function evaluations by inserting one row
    per iteration with a controlled sleep between inserts.
  - Thread B (reader): simulates OptimizerHistoryAccess by calling shutil.copy2()
    every 50 ms and then opening the copy with sqlite3.connect().

Run
---
    python tests/reproduce_copy_contention.py [--writes-per-second N] [--duration-secs S]

Expectations
------------
On a local SSD with DELETE journal mode you may not see errors at all, because
Windows sector-level atomicity protects page-1.  The test is most useful for:
  1. Demonstrating that the code path EXISTS and is in production.
  2. Confirming any fix (WAL mode, SQLite backup API, etc.) eliminates errors.
"""

import argparse
import shutil
import sqlite3
import tempfile
import threading
import time
import pathlib
import sys
import os

repo_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

DB_FILE = str(pathlib.Path(__file__).parent / "copy_contention_test.hst")

# ── writer ─────────────────────────────────────────────────────────────────────

def writer_thread(stop_event, writes_per_second=10):
    """
    Simulate pyOptSparse writing a .hst row for every IPOPT function eval.
    We do NOT use WAL mode here on purpose — that matches pyOptsparse's default.
    """
    sleep = 1.0 / writes_per_second

    con = sqlite3.connect(DB_FILE)
    con.execute("""
        CREATE TABLE IF NOT EXISTS iterations (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            major    INTEGER,
            obj      REAL,
            x_blob   BLOB
        )
    """)
    con.commit()

    major = 0
    while not stop_event.is_set():
        # Large blob simulates constraint/gradient data recorded per call
        blob = bytes(range(256)) * 200  # ~50 KB
        con.execute(
            "INSERT INTO iterations (major, obj, x_blob) VALUES (?, ?, ?)",
            (major, float(major) * 0.01, blob),
        )
        con.commit()
        major += 1
        time.sleep(sleep)

    con.close()
    print(f'[WRITER] Done. Wrote {major} rows.', flush=True)


# ── reader ─────────────────────────────────────────────────────────────────────

def reader_thread(stop_event, poll_interval=0.05):
    """
    Simulate OptimizerHistoryAccess._read_snapshot_copy():
      1. shutil.copy2() the live .hst to a temp file
      2. Open the copy with sqlite3.connect() and query row count
    """
    n_ok      = 0
    n_perm    = 0
    n_corrupt = 0
    n_other   = 0

    while not stop_event.is_set():
        if not os.path.exists(DB_FILE):
            time.sleep(0.01)
            continue

        with tempfile.NamedTemporaryFile(suffix=".hst", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Step 1: copy the live database.
            #   use_backup=True  → sqlite3.backup() (the fix)
            #   use_backup=False → shutil.copy2()   (the original racy code)
            if args.use_backup:
                with (
                    sqlite3.connect(DB_FILE, timeout=1.0) as src_con,
                    sqlite3.connect(tmp_path) as dst_con,
                ):
                    src_con.backup(dst_con, pages=-1)
            else:
                shutil.copy2(DB_FILE, tmp_path)

            # Step 2: open copy — this is where "malformed disk image" surfaces
            with sqlite3.connect(tmp_path, timeout=0.5) as con:
                count = con.execute("SELECT count(*) FROM iterations").fetchone()[0]
            n_ok += 1
            if n_ok % 100 == 0:
                print(f'[READER] {n_ok} successful copies (last count={count})', flush=True)

        except PermissionError as exc:
            n_perm += 1
            if n_perm <= 5 or n_perm % 50 == 0:
                print(f'[READER] PermissionError #{n_perm}: {exc}', flush=True)
        except sqlite3.DatabaseError as exc:
            msg = str(exc).lower()
            if 'malformed' in msg or 'corrupt' in msg:
                n_corrupt += 1
                print(f'[READER] CORRUPT/MALFORMED #{n_corrupt}: {exc}', flush=True)
            else:
                n_other += 1
                if n_other <= 5 or n_other % 20 == 0:
                    print(f'[READER] DB error #{n_other}: {exc}', flush=True)
        except Exception as exc:
            n_other += 1
            if n_other <= 5 or n_other % 20 == 0:
                print(f'[READER] Other error #{n_other}: {type(exc).__name__}: {exc}', flush=True)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        time.sleep(poll_interval)

    print(
        f'[READER] Done. ok={n_ok}, perm={n_perm}, corrupt={n_corrupt}, other={n_other}',
        flush=True,
    )
    return n_ok, n_perm, n_corrupt, n_other


# ── main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--writes-per-second', type=float, default=10,
                   help='Simulated IPOPT function eval rate (default: 10/s)')
    p.add_argument('--duration-secs', type=float, default=60,
                   help='How long to run the test in seconds (default: 60)')
    p.add_argument('--poll-interval', type=float, default=0.05,
                   help='Reader poll interval in seconds (default: 0.05)')
    p.add_argument('--use-backup', action='store_true', default=False,
                   help='Use sqlite3.backup() instead of shutil.copy2() (tests the fix)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Clean up from previous runs
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
    for suffix in ('-journal', '-wal', '-shm'):
        for f in [DB_FILE + suffix]:
            if os.path.exists(f):
                os.remove(f)

    stop_event = threading.Event()

    wt = threading.Thread(
        target=writer_thread,
        args=(stop_event, args.writes_per_second),
        daemon=True,
    )
    rt = threading.Thread(
        target=reader_thread,
        args=(stop_event, args.poll_interval),
        daemon=True,
    )

    print(
        f'Starting: {args.writes_per_second} writes/s, '
        f'{1/args.poll_interval:.0f} reads/s, '
        f'{args.duration_secs}s duration',
        flush=True,
    )

    wt.start()
    rt.start()

    time.sleep(args.duration_secs)
    stop_event.set()

    wt.join(timeout=5)
    rt.join(timeout=5)

    # Collect results from reader thread (stored in thread locals via join)
    # We just rely on the printed output for now.

    # Cleanup
    try:
        if os.path.exists(DB_FILE):
            os.remove(DB_FILE)
    except PermissionError as exc:
        print(f'[MAIN] Could not remove test db: {exc}', flush=True)

    print('\nTest complete.', flush=True)
    print('If you saw CORRUPT/MALFORMED lines above, the race is reproducible here.')
    print('If not, the race exists in the code but does not trigger on this hardware/OS.')
    print()
    print('The fix is to replace shutil.copy2() with SQLite\'s online backup API:')
    print('  src = sqlite3.connect(DB_FILE, timeout=1.0)')
    print('  dst = sqlite3.connect(copy_path)')
    print('  src.backup(dst)')
    print('  src.close(); dst.close()')
    print('This respects SQLite locking and always produces a consistent snapshot.')
