from pathlib import Path
import yaml

for f in sorted(Path("configs/crops").glob("*.yaml")):
    cfg = yaml.safe_load(f.read_text(encoding="utf-8"))
    for m in cfg["models"]:
        ok = Path(m["weights"]).exists()
        print(f"{cfg['id']:10} {m['id']:8} {'OK' if ok else 'THIEU FILE'}  {m['weights']}")