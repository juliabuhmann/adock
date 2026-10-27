"""Back-compat shim: test-access logging lives in src/adock/guard.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adock.guard import log_access  # noqa: F401
