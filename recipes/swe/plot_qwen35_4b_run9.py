# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Rebuild the single-row vector figure from the adjacent aggregate CSVs."""

import csv
from pathlib import Path
from xml.etree import ElementTree

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def main():
    directory = Path(__file__).resolve().parent
    with (directory / "qwen35_4b_run9_train.csv").open() as handle:
        training = list(csv.DictReader(handle))
    with (directory / "qwen35_4b_run9_validation.csv").open() as handle:
        validation = list(csv.DictReader(handle))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "svg.fonttype": "none",
            "svg.hashsalt": "qwen35_4b_run9",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3), layout="constrained")
    steps = [int(row["step"]) for row in training]
    metrics = [
        ("train_batch/rewards/mean", "Train reward", "#2563eb"),
        ("rollout_corr/k3_kl", "Rollout / update-start Actor k3 KL", "#7c3aed"),
    ]
    for ax, (key, title, color) in zip(axes, metrics, strict=False):
        values = [float(row[key]) for row in training]
        trailing = [sum(values[index - 9 : index + 1]) / 10 for index in range(9, len(values))]
        ax.plot(steps, values, color=color, alpha=0.35, linewidth=1, label="Per step")
        ax.plot(steps[9:], trailing, color=color, linewidth=2, label="Trailing 10 steps")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.legend(frameon=False, loc="upper left", fontsize=9)
    axes[0].set_ylim(0, 1)
    axes[1].set_ylim(bottom=0)
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))

    baseline, *checkpoints = validation
    ax = axes[2]
    ax.axhline(
        100 * float(baseline["mean_rm_score"]),
        color="#64748b",
        linestyle="--",
        label=f"Initial: {baseline['resolved']}/{baseline['num_scored_sessions']}",
    )
    ax.plot(
        [int(row["step"]) for row in checkpoints],
        [100 * float(row["mean_rm_score"]) for row in checkpoints],
        marker="o",
        color="#059669",
        linewidth=2,
        markersize=4,
        label="16 checkpoints",
    )
    ax.annotate(
        "Step 160: 282/500 (56.4%)",
        xy=(160, 56.4),
        xytext=(0.97, 0.94),
        textcoords="axes fraction",
        ha="right",
        fontsize=10,
        color="#047857",
    )
    ax.set_title("SWE-bench Verified validation", loc="left", fontweight="bold")
    ax.set_ylim(45, 59)
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    for ax in axes:
        ax.set_xlabel("Training step")
        ax.set_xlim(0, 164)
        ax.set_xticks([0, 40, 80, 120, 160])
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Qwen3.5-4B · Trial 450162305 / Run 9 · first 160 steps", fontsize=15)
    path = directory / "qwen35_4b_run9.svg"
    fig.savefig(path, metadata={"Date": None})
    plt.close(fig)
    ElementTree.register_namespace("", "http://www.w3.org/2000/svg")
    ElementTree.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    root = ElementTree.parse(path).getroot()
    root.insert(0, ElementTree.Element("{http://www.w3.org/2000/svg}title"))
    root[0].text = "Run 9: 160 training rewards and k3 KL values; 16 validation checkpoints"
    path.write_text(ElementTree.tostring(root, encoding="unicode").replace("\n", ""))


if __name__ == "__main__":
    main()
