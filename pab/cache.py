"""Prediction cache: one JSONL per (model, dataset, aug).

This is the backbone of the "compute on the GPU once, analyze locally forever"
workflow. A sweep appends predictions here; a killed or spot-preempted run resumes
by skipping ids already present. Scoring/plotting read these files offline.
"""
from __future__ import annotations
import json
import os
import re


def _slug(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(s)).strip("-")


def pred_path(results_dir, model_name, dataset_name, aug):
    fn = f"{_slug(model_name)}__{_slug(dataset_name)}__{_slug(aug)}.jsonl"
    return os.path.join(results_dir, "predictions", fn)


def load_done_ids(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    continue
    return done


def append(path, record):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
