"""Held-out test guard (preregistration amendment A2 test_guard).
guard(*items): items are set names or paths; if any refers to a test_* set / '/test_' path, exit unless
ADOCK_FINAL_EVAL=1, and when allowed append (timestamp, user, script, item, action) to results/test_access_log.tsv."""
import datetime, getpass, os, re, sys
from pathlib import Path

LOG = Path(__file__).resolve().parents[2] / "results/test_access_log.tsv"
_TEST = re.compile(r"(^|[/_.,\s])test_")


def is_test(item):
    return item is not None and any(_TEST.search(p) or p.startswith("test_") for p in str(item).split(","))


def log_access(script, item, action=""):
    new = not LOG.exists()
    with open(LOG, "a") as fh:
        if new:
            fh.write("timestamp_utc\tuser\tscript\tset\taction\n")
        fh.write(f"{datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}\t{getpass.getuser()}\t{script}\t{item}\t{action}\n")


def guard(*items, action=""):
    hit = [str(i) for i in items if is_test(i)]
    if not hit:
        return
    if os.environ.get("ADOCK_FINAL_EVAL") != "1":
        sys.exit(f"guard: refusing held-out test data {hit} (set ADOCK_FINAL_EVAL=1 only for the final evaluation)")
    for h in hit:
        log_access(Path(sys.argv[0]).name, h, action)
