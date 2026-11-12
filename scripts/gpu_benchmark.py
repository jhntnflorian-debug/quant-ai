"""Benchmark medido de matmul y entrenamiento; ejecutar en Colab, no en el PC."""
from __future__ import annotations

import argparse
import gc
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn

from quant_ai.config import load_config
from quant_ai.models.sequence import build_model
from quant_ai.training.trainer import amp_settings, environment_info, seed_everything, write_json

ROOT = Path(__file__).resolve().parents[1]


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def clean_memory(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def matmul_benchmark(device, mode, size, iterations):
    dtype = {"fp32": torch.float32, "tf32": torch.float32,
             "fp16": torch.float16, "bf16": torch.bfloat16}[mode]
    if device.type == "cpu" and mode != "fp32":
        return {"mode": mode, "status": "unsupported", "reason": "Smoke CPU solo mide fp32"}
    if mode == "tf32" and torch.cuda.get_device_capability(device)[0] < 8:
        return {"mode": mode, "status": "unsupported", "reason": "GPU sin TF32"}
    if mode == "bf16" and not torch.cuda.is_bf16_supported():
        return {"mode": mode, "status": "unsupported", "reason": "GPU sin bf16"}
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = mode == "tf32"
        a = torch.randn(size, size, device=device, dtype=dtype)
        b = torch.randn(size, size, device=device, dtype=dtype)
        c = torch.empty_like(a)
        with torch.no_grad():
            for _ in range(3):
                torch.mm(a, b, out=c)
            synchronize(device)
            start = time.perf_counter()
            for _ in range(iterations):
                torch.mm(a, b, out=c)
            synchronize(device)
        elapsed = time.perf_counter() - start
        return {"mode": mode, "status": "ok", "size": size, "iterations": iterations,
                "seconds": elapsed, "tflops": 2 * size ** 3 * iterations / elapsed / 1e12,
                "gpu_mem_peak_gb": torch.cuda.max_memory_allocated(device) / 2 ** 30
                if device.type == "cuda" else 0.0}
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def training_benchmark(device, name, batch, args, rows):
    dtype = amp_settings(device)
    model = build_model(name, args.n_features, hidden=args.hidden, layers=args.layers,
                        dropout=0.2, window=args.window).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    try:  # API vigente; torch.cuda.amp.GradScaler está obsoleta
        scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=dtype == torch.float16)
    x = torch.randn(batch, args.window, args.n_features, device=device)
    y = torch.randint(0, 2, (batch,), device=device).float()

    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
            loss = nn.functional.binary_cross_entropy_with_logits(model(x), y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        return loss.detach()

    for _ in range(2 if args.smoke else 5):
        step()
    synchronize(device)
    start = time.perf_counter()
    for _ in range(args.steps):
        loss = step()
    synchronize(device)
    elapsed = time.perf_counter() - start
    if not torch.isfinite(loss):
        raise FloatingPointError("Pérdida no finita durante el benchmark.")
    throughput = batch * args.steps / elapsed
    samples = {symbol: max(0, n - args.window + 1) for symbol, n in rows.items()}
    return {"model": name, "batch_size": batch, "status": "ok", "steps": args.steps,
            "seconds": elapsed, "samples_per_sec": throughput,
            "amp_dtype": str(dtype).removeprefix("torch.") if dtype else "disabled",
            "gpu_mem_peak_gb": torch.cuda.max_memory_allocated(device) / 2 ** 30
            if device.type == "cuda" else 0.0,
            "estimated_epoch_seconds_by_symbol": {s: n / throughput for s, n in samples.items()},
            "estimated_epoch_seconds": sum(samples.values()) / throughput if samples else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/base.yaml")
    parser.add_argument("--rows-per-symbol", nargs="*", default=[], metavar="BTCUSDT=60000",
                        help="Filas aproximadas reales por símbolo; sin esto no se estima una época.")
    parser.add_argument("--n-features", type=int, default=39)  # 39 = características reales por activo
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--matmul-size", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true", help="Prueba mínima explícitamente en CPU.")
    args = parser.parse_args()
    if min(args.n_features, args.window, args.hidden, args.layers, args.matmul_size, args.iterations, args.steps) < 1:
        parser.error("Dimensiones e iteraciones deben ser positivas.")
    try:
        rows = {symbol: int(n) for symbol, n in (item.split("=", 1) for item in args.rows_per_symbol)}
        if any(n < 1 for n in rows.values()):
            raise ValueError
    except ValueError:
        parser.error("Use --rows-per-symbol BTCUSDT=60000 ETHUSDT=60000 con cantidades positivas reales.")
    device = torch.device("cpu" if args.smoke else "cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and not args.smoke:
        parser.error("No hay GPU CUDA; use --smoke para una comprobación pequeña en CPU.")
    if args.smoke:
        args.window, args.hidden, args.layers = 8, 8, 1
        args.matmul_size, args.iterations, args.steps = 64, 2, 2
    seed_everything(args.seed)
    os.environ.setdefault("QUANT_AI_HOME", str(ROOT))
    cfg = load_config(args.config)
    report = {"created_at": datetime.now(timezone.utc).isoformat(),
              **environment_info(device, amp_settings(device)), "smoke": args.smoke,
              "shape": {k: getattr(args, k) for k in ("n_features", "window", "hidden", "layers")},
              "rows_per_symbol": rows, "seed": args.seed, "matmul": [], "training": [],
              "estimate_scope": "Solo ajuste con datos sintéticos residentes en GPU; excluye validación, "
                                "carga de datos, checkpoints y compilación. Filas antes de purga/huecos."}
    gpu = re.sub(r"[^A-Za-z0-9_-]+", "_", report["gpu_name"] or "CPU")
    date = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = cfg.reports_dir / "gpu_benchmark" / f"{gpu}_{date}.json"
    for mode in ("fp32", "tf32", "fp16", "bf16"):
        clean_memory(device)
        try:
            row = matmul_benchmark(device, mode, args.matmul_size, args.iterations)
        except torch.cuda.OutOfMemoryError:
            row = {"mode": mode, "status": "oom"}
        report["matmul"].append(row)
        write_json(path, report)
        print(row, flush=True)
    for name in ("gru", "tcn", "transformer"):
        for batch in ([4] if args.smoke else [256, 1024, 4096]):
            clean_memory(device)
            try:
                row = training_benchmark(device, name, batch, args, rows)
            except torch.cuda.OutOfMemoryError:
                row = {"model": name, "batch_size": batch, "status": "oom"}
            report["training"].append(row)
            write_json(path, report)
            print(row, flush=True)
    print(f"Benchmark guardado en {path}")


if __name__ == "__main__":
    main()
