"""Aggregate cached predictions into a leaderboard.

Reads results/predictions/*.jsonl and reports, per (model, dataset, aug): WER, CER,
real-time factor, latency p50/p95, and modeled self-host $/1000min. Runs entirely
offline on your laptop — no GPU, no re-transcription.

    python -m pab.report --gpu-price 0.40
"""
from __future__ import annotations
import argparse
import glob
import json
import os
from collections import defaultdict

from . import metrics
from .cost import self_host_cost_per_1000min


def load_records(results_dir):
    recs = []
    for p in glob.glob(os.path.join(results_dir, "predictions", "*.jsonl")):
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
    return recs


def aggregate(recs):
    groups = defaultdict(list)
    for r in recs:
        groups[(r["model"], r["dataset"], r["aug"])].append(r)
    rows = []
    for (model, dataset, aug), rs in groups.items():
        rtfs = [r["infer_sec"] / r["audio_sec"] for r in rs if r["audio_sec"] > 0]
        lat = [r["infer_sec"] for r in rs]
        rows.append({
            "model": model, "dataset": dataset, "aug": aug, "n": len(rs),
            "wer": metrics.corpus_rate(metrics.wer(r["ref"], r["hyp"]) for r in rs),
            "cer": metrics.corpus_rate(metrics.cer(r["ref"], r["hyp"]) for r in rs),
            "rtf": sum(rtfs) / len(rtfs) if rtfs else 0.0,
            "p50_s": metrics.percentile(lat, 50),
            "p95_s": metrics.percentile(lat, 95),
        })
    return rows


def to_markdown(rows, gpu_price_per_hr):
    hdr = ("| model | dataset | aug | n | WER% | CER% | RTF | p50 s | p95 s | "
           "$/1000min |\n|---|---|---|--:|--:|--:|--:|--:|--:|--:|\n")
    lines = []
    for r in sorted(rows, key=lambda x: (x["dataset"], x["aug"], x["wer"])):
        cost = self_host_cost_per_1000min(r["rtf"], gpu_price_per_hr)
        lines.append(
            f"| {r['model']} | {r['dataset']} | {r['aug']} | {r['n']} | "
            f"{r['wer'] * 100:.1f} | {r['cer'] * 100:.1f} | {r['rtf']:.3f} | "
            f"{r['p50_s']:.2f} | {r['p95_s']:.2f} | {cost:.2f} |")
    return hdr + "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--gpu-price", type=float, default=0.40,
                    help="$/hr for self-host cost model (RTX 4090 ~ $0.40)")
    args = ap.parse_args()

    recs = load_records(args.results_dir)
    if not recs:
        print("no predictions found — run `python -m pab.run --config ...` first")
        return
    md = to_markdown(aggregate(recs), args.gpu_price)
    out = os.path.join(args.results_dir, "leaderboard.md")
    with open(out, "w") as f:
        f.write(md)
    print(md)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
