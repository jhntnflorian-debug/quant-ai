"""Entrena modelos secuenciales en Colab y evalúa predicciones fuera de muestra."""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import pandas as pd
import torch

from quant_ai.config import load_config
from quant_ai.models.chart_cnn import ChartCNN
from quant_ai.data.store import load_all
import yaml
import numpy as np
from quant_ai.training.dataset import build_datasets
from quant_ai.training.chart_images import attach_image_cache, build_chart_folds, render_window
from quant_ai.training.trainer import (SequenceTrainer, amp_settings, environment_info,
                                       seed_everything, utc_now, write_json)

from evaluate_sequence import evaluate

ROOT = Path(__file__).resolve().parents[1]


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=["chart_cnn"], default="chart_cnn")
    for name, default in [("window", 48), ("epochs", 20), ("batch-size", 256), ("hidden", 64),
                          ("layers", 2), ("accum", 1), ("patience", 5), ("last-folds", 16),
                          ("num-workers", 4), ("seed", 42)]:
        ap.add_argument("--" + name, type=int, default=default)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--symbols", nargs="+")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/base.yaml")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    # Solo cambia la velocidad, no el experimento: por eso no entra en experiment.json.
    ap.add_argument("--no-cache", action="store_true",
                    help="renderizar cada imagen en cada época en lugar de pre-renderizarlas una vez por activo")
    return ap


def main():
    ap = parser()
    args = ap.parse_args()
    os.environ.setdefault("QUANT_AI_HOME", str(ROOT))
    v2 = yaml.safe_load((ROOT / "configs/experiments_v2.yaml").read_text(encoding="utf-8"))
    cfg = load_config(args.config, overrides={"labeling": {"method": "forward_net_return", "horizon_bars":24, "cost_threshold_bps":26}, "splits":{"walk_forward":{"first_test_start":v2["test_start"], "test_months":3}}})
    if args.window not in (48,96):
        ap.error("--window debe ser 48 o 96.")
    symbols = args.symbols or cfg.symbols
    if not symbols or not set(symbols).issubset(cfg.symbols) or len(symbols) != len(set(symbols)):
        ap.error("--symbols debe contener símbolos únicos de la configuración.")
    if args.smoke:
        symbols, args.last_folds, args.epochs = symbols[:1], 1, 1
    hp = dict(window=args.window)
    out = args.out or cfg.models_dir / "dl" / f"chart_cnn_w{args.window}"
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"model": args.model, "hyperparams": hp, "epochs": args.epochs,
                "batch_size": args.batch_size, "lr": args.lr, "accum": args.accum, "clip": args.clip,
                "patience": args.patience, "last_folds": args.last_folds, "symbols": symbols,
                "seed": args.seed, "smoke": args.smoke, "config": cfg.raw}
    manifest_path = out / "experiment.json"
    if manifest_path.exists():
        if not args.resume:
            ap.error("La salida ya existe; use --resume u otra carpeta.")
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            ap.error("La configuración cambió. Use otra carpeta para este experimento.")
    write_json(manifest_path, manifest)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    previous = json.loads((out / "run_info.json").read_text(encoding="utf-8")) if (out / "run_info.json").exists() else {}
    info = {"model": args.model, "hyperparams": {**hp, **{k: manifest[k] for k in
            ("epochs", "batch_size", "lr", "accum", "clip", "patience", "seed", "last_folds")}},
            "n_params": None, **environment_info(device, amp_settings(device)), "status": "running",
            "started_at": previous.get("started_at", utc_now()), "finished_at": None,
            "total_seconds": previous.get("total_seconds", 0.0), "folds": [], "smoke": args.smoke}
    write_json(out / "run_info.json", info)
    started, prior_seconds = time.perf_counter(), info["total_seconds"]
    predictions = []
    try:
        # Se construyen ambos activos para preservar las características cruzadas.
        raw = load_all(cfg)
        raw = {s: df.loc[df.open_time < cfg.final_holdout_start].copy() for s,df in raw.items()}
        datasets = build_datasets(cfg, final=False, raw=raw)
        for symbol in symbols:
            folds = list(build_chart_folds(cfg, datasets[symbol], args.window, args.last_folds, args.smoke))
            if not args.no_cache:
                # La imagen depende solo de (activo, ventana, posición final): se renderiza una vez en RAM y
                # la comparten todos los folds y épocas. Salida idéntica a renderizar en __getitem__.
                cache_started = time.perf_counter()
                n_images, n_bytes = attach_image_cache(
                    (part for f in folds for part in (f.fit, f.validation, f.calibration, f.test)),
                    workers=os.cpu_count() or 1)
                print(f"{symbol}: {n_images} imágenes pre-renderizadas en {time.perf_counter() - cache_started:.1f} s "
                      f"({n_bytes / 2**30:.2f} GiB en RAM, {os.cpu_count() or 1} procesos)", flush=True)
            for fold in folds:
                fold_id = f"{symbol}:{fold.fold.k}"
                fold_out = out / "folds" / symbol / f"fold_{fold.fold.k:03d}"
                seed_everything(args.seed + fold.fold.k)
                model = ChartCNN(**hp)
                samples = out / "samples"
                samples.mkdir(exist_ok=True)
                if not list(samples.glob("*.npy")):
                    for i in range(min(16,len(fold.fit))):
                        end = fold.fit.positions[i]
                        np.save(samples / f"sample_{i:02d}.npy", render_window(*fold.fit.values[end-args.window+1:end+1].T))
                trainer = SequenceTrainer(model, fold_out, model_name=args.model,
                    hyperparams={**hp, "feature_names": fold.feature_names}, epochs=args.epochs,
                    batch_size=args.batch_size, lr=args.lr, accum=args.accum, clip=args.clip,
                    patience=args.patience, num_workers=args.num_workers, seed=args.seed + fold.fold.k,
                    compile_model=args.compile, fold=fold_id, aggregate_metrics=out / "metrics.csv")
                info["n_params"] = trainer.info["n_params"]
                item = {"symbol": symbol, "fold": fold.fold.k, "test_start": fold.fold.test_start.isoformat(),
                        "test_end": fold.fold.test_end.isoformat(), "n_fit": len(fold.fit),
                        "n_calibration": len(fold.calibration), "n_test": len(fold.test),
                        "checkpoint_dir": str(fold_out / "checkpoints"), "status": "running"}
                info["folds"].append(item)
                write_json(out / "run_info.json", info)
                trainer.fit(fold.fit, fold.validation, normalization=fold.normalization, resume=args.resume)
                pred = trainer.calibrate_predict(fold, datasets[symbol].labels)
                pred.to_parquet(fold_out / "predictions.parquet")
                predictions.append(pred)
                item["status"] = "finished"
                item["best_val_loss"] = trainer.best_metric
                info["total_seconds"] = prior_seconds + time.perf_counter() - started
                write_json(out / "run_info.json", info)
                del trainer, model, fold
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            del folds  # libera la caché de este activo antes de construir la del siguiente
            gc.collect()
        result = pd.concat(predictions).sort_index()
        temporary = out / "predictions.tmp.parquet"
        result.to_parquet(temporary)
        os.replace(temporary, out / "predictions.parquet")
        evaluate(out / "predictions.parquet", f"dl_{args.model}", cfg=cfg, datasets=datasets)
        info["status"] = "finished"
    except KeyboardInterrupt:
        info["status"] = "interrupted"
        raise
    except BaseException:
        info["status"] = "failed"
        raise
    finally:
        info["total_seconds"] = prior_seconds + time.perf_counter() - started
        info["finished_at"] = utc_now()
        for item in info["folds"]:
            if item["status"] == "running" and info["status"] in ("failed", "interrupted"):
                item["status"] = info["status"]
        write_json(out / "run_info.json", info)
    print(f"Artefactos guardados en {out}")


if __name__ == "__main__":
    main()
