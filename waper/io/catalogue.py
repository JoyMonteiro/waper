import json
import pandas as pd
from pathlib import Path
from . import extract

def write_meta(path, meta: dict) -> None:
    path = Path(path); path.mkdir(parents=True, exist_ok=True)
    (path / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

def read_meta(path) -> dict:
    return json.loads((Path(path) / "meta.json").read_text())

_TABLES = {
    "nodes": extract.extract_nodes,
    "edges": extract.extract_edges,
    "rwps": extract.extract_rwps,
    "samples": extract.extract_samples,
    "track_nodes": extract.extract_track_nodes,
    "track_edges": extract.extract_track_edges,
}

def _write_table(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "part.parquet", engine="pyarrow", index=False)

def save_catalogue(waper, path, *, meta=None) -> None:
    path = Path(path); path.mkdir(parents=True, exist_ok=True)
    for name, fn in _TABLES.items():
        _write_table(fn(waper), path / name)
    write_meta(path, meta or {})

class Catalogue:
    def __init__(self, path):
        self.path = Path(path)
        self.meta = read_meta(self.path) if (self.path/"meta.json").exists() else {}
        self._cache = {}

    def table(self, name: str) -> pd.DataFrame:
        if name not in self._cache:
            files = sorted((self.path / name).glob("*.parquet"))
            if not files:
                raise FileNotFoundError(f"no parquet for table {name!r} in {self.path}")
            self._cache[name] = pd.concat(
                (pd.read_parquet(f, engine="pyarrow") for f in files), ignore_index=True)
        return self._cache[name]

def load_catalogue(path) -> "Catalogue":
    return Catalogue(path)
