"""Run a benchmark sweep.

For each (model x dataset x augmentation), transcribe every sample and append
predictions to a JSONL cache. Idempotent + resumable: already-transcribed samples
are skipped, so a preempted run just continues.

    python -m pab.run --config configs/smoke.yaml
"""
from __future__ import annotations
import argparse
import yaml

from . import cache
from .asr import build_backend
from .datasets import load_dataset_spec


def _resolve_augs(augs):
    if list(augs) == ["none"]:
        return {"none": lambda a, sr: a}
    from .telephony import AUGS
    return AUGS


def run(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    results_dir = cfg.get("results_dir", "results")
    augs = cfg.get("augmentations", ["none"])
    aug_fns = _resolve_augs(augs)

    for mspec in cfg["models"]:
        mname = mspec["name"]
        print(f"\n=== model: {mname} ===")
        backend = build_backend(mspec)
        for dspec in cfg["datasets"]:
            dname = dspec["name"]
            wlang = dspec.get("whisper_lang")
            # load samples once per dataset, reuse across augmentations
            samples = list(load_dataset_spec(dspec))
            for aug in augs:
                path = cache.pred_path(results_dir, mname, dname, aug)
                done = cache.load_done_ids(path)
                n_new = 0
                for s in samples:
                    if s["id"] in done:
                        continue
                    audio = aug_fns[aug](s["audio"], s["sr"])
                    tr = backend.transcribe(audio, s["sr"], language=wlang)
                    cache.append(path, {
                        "id": s["id"], "model": mname, "dataset": dname,
                        "aug": aug, "language": s["language"],
                        "ref": s["ref"], "hyp": tr.text,
                        "audio_sec": round(len(audio) / s["sr"], 4),
                        "infer_sec": round(tr.infer_sec, 4),
                    })
                    n_new += 1
                print(f"  {dname} [{aug}]: +{n_new} new "
                      f"(cached {len(done) + n_new}) -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    run(ap.parse_args().config)


if __name__ == "__main__":
    main()
