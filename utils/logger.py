"""Per-session CSV logger for RT-MBAS feature vectors."""
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from app.config import DATA_PATH, SESSIONS_PATH


class SessionLogger:
    """
    Logs per-frame feature dicts to two CSV files:
      - data/dataset.csv        (cumulative across all sessions)
      - data/sessions/<id>.csv  (this session only)
    """

    def __init__(self):
        """Generate a unique session ID and prepare file paths."""
        self.session_id: str = datetime.now().strftime("%Y%m%d%H%M%S")
        SESSIONS_PATH.mkdir(parents=True, exist_ok=True)
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._session_path = SESSIONS_PATH / f"{self.session_id}.csv"
        self._dataset_has_header = DATA_PATH.exists() and DATA_PATH.stat().st_size > 0
        self._session_has_header = False

    def log_frame(self, feature_dict: dict) -> None:
        """
        Append one feature row to both dataset.csv and the session CSV.

        Creates headers automatically on the first write to each file.
        """
        df = pd.DataFrame([feature_dict])

        # ── cumulative dataset ────────────────────────────────────────────────
        df.to_csv(
            DATA_PATH,
            mode="a",
            index=False,
            header=not self._dataset_has_header,
        )
        self._dataset_has_header = True

        # ── per-session file ──────────────────────────────────────────────────
        df.to_csv(
            self._session_path,
            mode="a",
            index=False,
            header=not self._session_has_header,
        )
        self._session_has_header = True

    def close(self) -> None:
        """Finalize the session (pandas flushes on every write; nothing extra needed)."""
        pass
