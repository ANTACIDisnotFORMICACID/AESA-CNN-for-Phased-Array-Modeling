from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "v2.12" / "src"))
sys.path.insert(0, str(ROOT.parent / "v2.10" / "src"))
