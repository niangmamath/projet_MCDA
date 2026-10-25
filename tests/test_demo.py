import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_demo_runs():
    """La démonstration du README s'exécute de bout en bout et produit le rapport PDF."""
    out = subprocess.run([sys.executable, str(ROOT / "examples" / "demo.py")], capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    assert "MedTech Nova + Agence nationale" in out.stdout
    assert "même classement : True" in out.stdout
    assert (ROOT / "demo_output" / "rapport_stratema.pdf").read_bytes().startswith(b"%PDF")
