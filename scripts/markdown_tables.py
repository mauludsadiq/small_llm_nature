from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    csv_path = ROOT / "out_k_sweep.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}. Run: python scripts/export_k_sweep.py")

    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            row["repeat"] = int(row["repeat"])
            row["k"] = int(row["k"])
            row["H_test"] = float(row["H_test"])
            rows.append(row)

    Ns = sorted({row["repeat"] for row in rows})
    Ks = sorted({row["k"] for row in rows})

    grid = defaultdict(dict)
    for row in rows:
        grid[row["repeat"]][row["k"]] = row["H_test"]

    md = []
    md.append("# H_test grid")
    md.append("")
    header = ["N (repeat)"] + [f"H_test(k={k})" for k in Ks]
    md.append("| " + " | ".join(header) + " |")
    md.append("| " + " | ".join(["---"] * len(header)) + " |")
    for N in Ns:
        vals = [f"{grid[N].get(k, float('nan')):.4f}" for k in Ks]
        md.append("| " + " | ".join([str(N)] + vals) + " |")
    md.append("")

    out_path = ROOT / "out_tables.md"
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
