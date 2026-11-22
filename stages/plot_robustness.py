"""Plot one stage-4 matrix plan. Usage: python stages/plot_robustness.py runs/scan"""
import argparse
from collections import defaultdict
import json

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(out):
    rows = [json.loads(p.read_text()) for p in sorted((out / "rows").glob("*.json"))]
    if not rows:
        raise ValueError("No measurement rows")
    positives = [r for r in rows if not r["negative"]]
    for prefix, label in (("data", "SFT dataset (%)"), ("prune", "Pruned weights (%)"),
                          ("cpt", "CPT input-token budget")):
        series = defaultdict(list)
        for r in positives:
            suffix = r["case"].removeprefix(prefix)
            if r["terminal"] and r["case"].startswith(prefix) and suffix.isdigit():
                series[int(suffix)].append(r)
        if not series:
            continue
        x = sorted(series)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, names in zip(axes, [("survival_rate","joint_pass"), ("ppl_ratio","top1")]):
            for key in names:
                ax.plot(x, [sum(r[key] for r in series[n])/len(series[n]) for n in x],
                        marker="o", label=key)
            ax.set_xlabel(label)
            if prefix == "cpt":
                ax.set_xscale("log")
            ax.legend()
            ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(out / f"{prefix}-strength.png", dpi=160)
        plt.close(fig)
    combos = sorted({r["case"] for r in positives if r["case"].startswith("C")})
    if combos:
        fig, ax = plt.subplots(figsize=(8, 4))
        for case in combos:
            group = defaultdict(list)
            for r in positives:
                if r["case"] == case:
                    group[r["step"]].append(r["survival_rate"])
            steps = sorted(group)
            ax.plot(steps, [sum(group[n])/len(group[n]) for n in steps], marker="o", label=case)
        ax.set(xlabel="Transformation step", ylabel="Mean retained bit fraction", xticks=[1,2,3])
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "combination-steps.png", dpi=160)
        plt.close(fig)
    terminal = [r for r in rows if r["terminal"]]
    if terminal:
        labels = sorted({r["source"] for r in terminal})
        decisions = ["U1","U2","U3","blocked","no_hit","multiple"]
        counts = [[sum(r["source"]==src and r["decision"]==d for r in terminal)
                   for d in decisions] for src in labels]
        fig, ax = plt.subplots(figsize=(8,4))
        ax.imshow(counts, cmap="Blues", aspect="auto")
        for i, row in enumerate(counts):
            for j, count in enumerate(row):
                ax.text(j, i, str(count), ha="center", va="center",
                        color="white" if count > max(map(max, counts))*.6 else "black")
        ax.set(xticks=range(len(decisions)), xticklabels=decisions,
               yticks=range(len(labels)), yticklabels=labels,
               xlabel="Decision (pooled terminal cases)", ylabel="Source")
        fig.tight_layout()
        fig.savefig(out / "confusion.png",dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    plot(parser.parse_args().output)
