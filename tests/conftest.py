"""pytest setup: every test run gets its own empty data folder, so tests never read or change data/."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("DADS_DATA_DIR", tempfile.mkdtemp(prefix="dads_test_data_"))
os.environ.setdefault("DADS_NO_BROWSER", "1")
# The first account the server creates in that empty folder (see src/storage/auth_store.py).
os.environ.setdefault("DADS_ADMIN_USER", "admin")
os.environ.setdefault("DADS_ADMIN_PASSWORD", "test-only-password-123")
