import os, json
import numpy as np

data_dir = r"E:\mattergen\data"
vals = []
bad = []
for fn in os.listdir(data_dir):
    if not (fn.startswith("JVASP-") and fn.endswith(".json")):
        continue
    p = os.path.join(data_dir, fn)
    with open(p, "r", encoding="utf-8") as f:
        d = json.load(f)
    props = d.get("properties", {})
    vals.append([
        float(props.get("deltaG_H", 0.0)),
        float(props.get("thermo_stability", 0.0)),
        float(props.get("synth_score", 0.5)),
    ])

vals = np.array(vals)
print("N =", len(vals))
for i, name in enumerate(["deltaG_H", "thermo_stability", "synth_score"]):
    v = vals[:, i]
    print(name, "min/mean/max/std =", v.min(), v.mean(), v.max(), v.std())
