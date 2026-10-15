from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
for version in (ROOT, ROOT.parent / "v2.14", ROOT.parent / "v2.12", ROOT.parent / "v2.10"):
    sys.path.insert(0, str(version / "src"))
