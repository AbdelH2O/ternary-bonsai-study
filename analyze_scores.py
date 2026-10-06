"""Compute Bonsai 2 versus Qwen3.8 deltas from whitepaper Table 10 and 11."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main():
    with (ROOT / "published_scores.csv").open(newline="") as file:
        scores = list(csv.DictReader(file))
    totals = defaultdict(lambda: defaultdict(list))
    output = []
    for row in scores:
        item = {"category": row["category"], "benchmark": row["benchmark"]}
        for effort in ("xhigh", "medium"):
            qwen = float(row[f"qwen_{effort}"])
            bonsai = float(row[f"bonsai_{effort}"])
            delta = bonsai - qwen
            item[f"delta_{effort}_points"] = round(delta, 2)
            totals[row["category"]][effort].append(delta)
        output.append(item)
    (ROOT / "results").mkdir(parents=True, exist_ok=True)
    with (ROOT / "results" / "benchmark_deltas.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print("Category mean Bonsai minus Qwen, percentage points:")
    for category, efforts in totals.items():
        print(f"  {category:25} xhigh={sum(efforts['xhigh'])/len(efforts['xhigh']):+5.2f} "
              f"medium={sum(efforts['medium'])/len(efforts['medium']):+5.2f}")
    for effort in ("xhigh", "medium"):
        ranked = sorted(output, key=lambda row: row[f"delta_{effort}_points"])
        print(f"Largest {effort} gaps:")
        for row in ranked[:5]:
            print(f"  {row['benchmark']:22} {row[f'delta_{effort}_points']:+.2f}")


if __name__ == "__main__":
    main()
