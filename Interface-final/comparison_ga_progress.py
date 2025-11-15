# comparison_ga_progress.py
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List

from utils import load_wdbc, outer_split, clf_pipeline, compute_metrics, jaccard
from genetic_algorithm import GeneticAlgorithmFS, GAConfig

OUTDIR = Path("results")
OUTDIR.mkdir(exist_ok=True, parents=True)

def run_once(seed: int, ga_cfg_overrides=None):
    X, y, feat_names = load_wdbc()
    X_tr, X_te, y_tr, y_te = outer_split(X, y, test_size=0.3, seed=seed)

    cfg = GAConfig(random_state=seed)
    if ga_cfg_overrides:
        for k, v in ga_cfg_overrides.items():
            setattr(cfg, k, v)

    ga = GeneticAlgorithmFS(X_tr, y_tr, cfg)
    mask, fit, hist = ga.run()
    # تقييم خارجي على مجموعة الاختبار
    idx = np.where(mask == 1)[0]
    if idx.size == 0:
        idx = np.array([0])
    model = clf_pipeline()
    model.fit(X_tr[:, idx], y_tr)
    y_pred = model.predict(X_te[:, idx])
    y_prob = model.predict_proba(X_te[:, idx])[:, 1]
    m = compute_metrics(y_te, y_prob, y_pred)

    return {
        "seed": seed,
        "selected_k": int(mask.sum()),
        "selected_names": [feat_names[i] for i in idx],
        "fitness": float(fit),
        "history": hist,
        "metrics": m.__dict__,
        "mask": mask.tolist()
    }

def main(n_runs=5):
    results = []
    masks = []
    for s in range(n_runs):
        r = run_once(seed=42 + s, ga_cfg_overrides=None)
        results.append(r)
        masks.append(r["mask"])
    # استقرار الاختيار (Jaccard متوسط)
    jac = []
    for i in range(len(masks)):
        for j in range(i+1, len(masks)):
            jac.append(jaccard(masks[i], masks[j]))
    stab = float(np.mean(jac)) if jac else 1.0

    Path(OUTDIR).mkdir(parents=True, exist_ok=True)
    with open(OUTDIR / "ga_runs.json", "w", encoding="utf-8") as f:
        json.dump({"runs": results, "stability_jaccard_mean": stab}, f, indent=2)
    print(f"Saved runs to {OUTDIR/'ga_runs.json'} | Jaccard mean={stab:.3f}")

if __name__ == "__main__":
    main(n_runs=5)
