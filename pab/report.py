"""Aggregate cached predictions into a leaderboard, with uncertainty.

Reads results/predictions/*.jsonl and reports, per (model, dataset, aug): WER (with a
bootstrap 95% CI), CER, real-time factor, latency p50/p95, and modeled self-host
$/1000min. Runs entirely offline on your laptop — no GPU, no re-transcription.

    python -m pab.report --gpu-price 0.69
    python -m pab.report --compare          # paired A/B tests between models

Use --compare for any "model A beats model B" claim: it pairs the systems on identical
utterances, which is both the correct test and far more sensitive than comparing the
marginal CIs in the main table.
"""
from __future__ import annotations
import argparse
import glob
import itertools
import json
import os
from collections import defaultdict

from . import metrics
from .cost import self_host_cost_per_1000min
from .stats import bootstrap_ci, paired_bootstrap


def load_records(results_dir):
    recs = []
    for p in glob.glob(os.path.join(results_dir, "predictions", "*.jsonl")):
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
    return recs


def group(recs):
    """(model, dataset, aug) -> list of records."""
    g = defaultdict(list)
    for r in recs:
        g[(r["model"], r["dataset"], r["aug"])].append(r)
    return g


def aggregate(groups, n_boot=2000):
    rows = []
    for (model, dataset, aug), rs in groups.items():
        wer_pairs = [metrics.wer(r["ref"], r["hyp"]) for r in rs]
        cer_pairs = [metrics.cer(r["ref"], r["hyp"]) for r in rs]
        rtfs = [r["infer_sec"] / r["audio_sec"] for r in rs if r["audio_sec"] > 0]
        lat = [r["infer_sec"] for r in rs]
        wer, lo, hi = bootstrap_ci(wer_pairs, n_boot=n_boot)
        rows.append({
            "model": model, "dataset": dataset, "aug": aug, "n": len(rs),
            "wer": wer, "wer_lo": lo, "wer_hi": hi,
            "cer": metrics.corpus_rate(cer_pairs),
            "rtf": sum(rtfs) / len(rtfs) if rtfs else 0.0,
            "p50_s": metrics.percentile(lat, 50),
            "p95_s": metrics.percentile(lat, 95),
        })
    return rows


def to_markdown(rows, gpu_price_per_hr):
    hdr = ("| model | dataset | aug | n | WER% (95% CI) | CER% | RTF | p50 s | p95 s "
           "| $/1000min |\n|---|---|---|--:|:--|--:|--:|--:|--:|--:|\n")
    lines = []
    for r in sorted(rows, key=lambda x: (x["dataset"], x["aug"], x["wer"])):
        cost = self_host_cost_per_1000min(r["rtf"], gpu_price_per_hr)
        ci = f"{r['wer']*100:.1f} [{r['wer_lo']*100:.1f}–{r['wer_hi']*100:.1f}]"
        lines.append(
            f"| {r['model']} | {r['dataset']} | {r['aug']} | {r['n']} | {ci} | "
            f"{r['cer']*100:.1f} | {r['rtf']:.3f} | {r['p50_s']:.2f} | "
            f"{r['p95_s']:.2f} | {cost:.2f} |")
    return hdr + "\n".join(lines) + "\n"


def compare(groups, n_boot=2000, alpha=0.05):
    """Paired A/B between every model pair sharing a (dataset, aug) and utterance set."""
    by_cell = defaultdict(dict)          # (dataset, aug) -> model -> {id: record}
    for (model, dataset, aug), rs in groups.items():
        by_cell[(dataset, aug)][model] = {r["id"]: r for r in rs}

    out = []
    for (dataset, aug), models in sorted(by_cell.items()):
        for a, b in itertools.combinations(sorted(models), 2):
            ids = sorted(set(models[a]) & set(models[b]))
            if len(ids) < 2:
                continue
            ap = [metrics.wer(models[a][i]["ref"], models[a][i]["hyp"]) for i in ids]
            bp = [metrics.wer(models[b][i]["ref"], models[b][i]["hyp"]) for i in ids]
            diff, lo, hi, p = paired_bootstrap(ap, bp, n_boot=n_boot, alpha=alpha)
            out.append({
                "dataset": dataset, "aug": aug, "a": a, "b": b, "n": len(ids),
                "diff": diff, "lo": lo, "hi": hi, "p": p,
                "significant": not (lo <= 0.0 <= hi),
            })
    return out


def compare_markdown(cmps):
    hdr = ("\n## Paired A/B (WER difference, same utterances)\n\n"
           "| dataset | aug | n | A | B | ΔWER pp (A−B) | 95% CI | p | verdict |\n"
           "|---|---|--:|---|---|--:|:--|--:|---|\n")
    lines = []
    for c in cmps:
        verdict = "**significant**" if c["significant"] else "not significant"
        lines.append(
            f"| {c['dataset']} | {c['aug']} | {c['n']} | {c['a']} | {c['b']} | "
            f"{c['diff']*100:+.2f} | [{c['lo']*100:+.2f}, {c['hi']*100:+.2f}] | "
            f"{c['p']:.3f} | {verdict} |")
    return hdr + "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--gpu-price", type=float, default=0.40,
                    help="$/hr for self-host cost model")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap replicates")
    ap.add_argument("--compare", action="store_true",
                    help="also run paired A/B tests between models")
    args = ap.parse_args()

    recs = load_records(args.results_dir)
    if not recs:
        print("no predictions found — run `python -m pab.run --config ...` first")
        return
    groups = group(recs)
    md = to_markdown(aggregate(groups, args.boot), args.gpu_price)
    if args.compare:
        md += compare_markdown(compare(groups, args.boot))
    out = os.path.join(args.results_dir, "leaderboard.md")
    with open(out, "w") as f:
        f.write(md)
    print(md)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
