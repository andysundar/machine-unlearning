
import os, random, numpy as np, torch

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- CSV logger --------------------------------------------------------------

class CSVLogger:
    def __init__(self, path, header=None, fieldnames=None, mode="a"):
        """
        Accepts both `header=` and `fieldnames=` for compatibility.
        """
        self.path = path
        self.header = header or fieldnames or []
        self.mode = mode
        # create parent dir if needed
        if self.path:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._ensure_header()

    def _ensure_header(self):
        if not self.path:
            return
        need_header = (not os.path.exists(self.path)) or os.path.getsize(self.path) == 0
        if need_header and self.header:
            with open(self.path, "w", encoding="utf-8") as f:
                f.write(",".join(self.header) + "\n")

    def log(self, row):
        """
        row: dict (preferred) or list/tuple aligned to header
        """
        if not self.path:
            return
        if isinstance(row, dict):
            vals = [str(row.get(k, "")) for k in self.header]
        else:
            vals = [str(x) for x in row]
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(",".join(vals) + "\n")

    # Optional context-manager + no-op close so callers can safely close()
    def close(self):
        # nothing persistent to close; kept for API compatibility
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()



# --- device selection helper ---

def device_pick(prefer: str = "auto") -> torch.device:
    """
    Choose a good default device.
    prefer: "auto" | "cuda" | "mps" | "cpu"
    - "auto": cuda > mps (Apple Silicon) > cpu
    """
    prefer = (prefer or "auto").lower()

    if prefer in ("cuda", "gpu"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        # fallback to auto
    if prefer == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        # fallback to auto
    if prefer == "cpu":
        return torch.device("cpu")

    # auto detection
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
