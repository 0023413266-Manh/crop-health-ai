from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
for f in sorted((ROOT / "configs/crops").glob("*.yaml")):
    cfg = yaml.safe_load(f.read_text(encoding="utf-8"))
    p = ROOT / cfg["knowledge_file"]
    print(f"{cfg['id']:10} {'OK' if p.exists() else 'THIEU FILE'}  {cfg['knowledge_file']}")