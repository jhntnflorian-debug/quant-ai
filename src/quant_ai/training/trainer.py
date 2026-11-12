"""Entrenamiento secuencial en CUDA/CPU; TPU no está soportado en esta versión."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, RandomSampler

METRIC_COLUMNS = ["epoch", "fold", "train_loss", "val_loss", "val_auc", "val_brier", "lr",
                  "epoch_seconds", "samples_per_sec", "gpu_mem_alloc_gb", "gpu_mem_peak_gb"]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(_worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def amp_settings(device):
    if device.type != "cuda":
        return None
    # bf16 solo con soporte nativo (sm_80 o superior: A100, L4, H100). En una T4 (sm_75) algunas versiones
    # de torch dicen que bf16 está "soportado" por emulación, que es lenta: ahí se usa fp16 con GradScaler.
    native_bf16 = torch.cuda.get_device_capability(device)[0] >= 8 and torch.cuda.is_bf16_supported()
    return torch.bfloat16 if native_bf16 else torch.float16


def environment_info(device, dtype):
    cuda = device.type == "cuda"
    return {"device": str(device), "gpu_name": torch.cuda.get_device_name(device) if cuda else None,
            "gpu_vram_gb": torch.cuda.get_device_properties(device).total_memory / 2 ** 30 if cuda else None,
            "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "amp_dtype": str(dtype).removeprefix("torch.") if dtype else "disabled"}


def dataset_signature(*datasets):
    """Detecta cambios de datos al reanudar, sin materializar ventanas."""
    digest = hashlib.sha256()
    seen = set()
    for ds in datasets:
        if id(ds.values) not in seen:
            digest.update(np.ascontiguousarray(ds.values).view(np.uint8))
            seen.add(id(ds.values))
        digest.update(np.ascontiguousarray(ds.labels).view(np.uint8))
        digest.update(ds.index.asi8.tobytes())
        digest.update(ds.positions.tobytes())
        digest.update(str(ds.window).encode())
    return digest.hexdigest()


class SequenceTrainer:
    """Un entrenador por símbolo/fold; out contiene checkpoints, CSV y run_info.

    Se reanuda desde la última época completa. No se permite cambiar el horizonte
    de OneCycle al reanudar. torch.compile es opcional y está apagado por defecto.
    """
    def __init__(self, model, out, *, model_name, hyperparams, epochs=30, batch_size=512,
                 lr=1e-3, accum=1, clip=1.0, patience=5, num_workers=2, seed=42,
                 compile_model=False, device=None, fold=0, aggregate_metrics=None):
        if min(epochs, batch_size, accum, patience) < 1 or lr <= 0 or clip <= 0 or num_workers < 0:
            raise ValueError("Parámetros de entrenamiento fuera de rango.")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type not in ("cuda", "cpu"):
            raise ValueError("Solo se admiten CUDA y CPU; TPU no está soportado.")
        self.dtype = amp_settings(self.device)
        self.model = model.to(self.device)
        self.network = torch.compile(self.model) if compile_model else self.model
        self.out = Path(out)
        self.aggregate_metrics = Path(aggregate_metrics) if aggregate_metrics else None
        (self.out / "checkpoints").mkdir(parents=True, exist_ok=True)
        self.epochs, self.batch_size, self.lr = epochs, batch_size, lr
        self.accum, self.clip, self.patience = accum, clip, patience
        self.num_workers, self.seed, self.fold = num_workers, seed, fold
        self.model_name, self.hyperparams = model_name, hyperparams
        self.settings = dict(epochs=epochs, batch_size=batch_size, lr=lr, accum=accum,
                             clip=clip, patience=patience, seed=seed)
        self.generator = torch.Generator().manual_seed(seed)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        # torch.amp.GradScaler es la API vigente (torch >= 2.3); torch.cuda.amp.GradScaler está obsoleta.
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.dtype == torch.float16)
        except (AttributeError, TypeError):  # torch antiguo
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.dtype == torch.float16)
        self.scheduler = None
        self.epoch, self.best_metric, self.bad_epochs = 0, math.inf, 0
        self.history, self.normalization = [], {}
        self.signature = None
        self.info = {"model": model_name, "hyperparams": {**hyperparams, **self.settings},
                     "n_params": sum(p.numel() for p in model.parameters()),
                     **environment_info(self.device, self.dtype), "status": "running",
                     "started_at": utc_now(), "finished_at": None, "total_seconds": 0.0,
                     "folds": [fold]}

    def prepare(self, n_samples):
        steps = math.ceil(math.ceil(n_samples / self.batch_size) / self.accum)
        if steps < 1:
            raise ValueError("El conjunto de ajuste está vacío.")
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.lr, total_steps=steps * self.epochs)

    def loader(self, dataset, shuffle=False):
        # La creación de workers no debe consumir el RNG del orden de muestras:
        # persistent_workers crea procesos una sola vez; resume los crea de nuevo.
        sampler = RandomSampler(dataset, generator=self.generator) if shuffle else None
        worker_generator = torch.Generator().manual_seed(self.seed)
        return DataLoader(dataset, batch_size=self.batch_size, sampler=sampler,
                          num_workers=self.num_workers, pin_memory=self.device.type == "cuda",
                          persistent_workers=self.num_workers > 0, generator=worker_generator,
                          worker_init_fn=seed_worker)

    def autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.dtype is not None)

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _train_epoch(self, loader):
        self.network.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss_sum, seen = 0.0, 0
        self.sync()
        started = time.perf_counter()
        for step, (x, y) in enumerate(loader):
            if step % self.accum == 0:
                group_size = min(self.accum * self.batch_size, len(loader.dataset) - seen)
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            with self.autocast():
                logits = self.network(x)
                summed = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="sum")
                loss = summed / group_size
            if not torch.isfinite(loss):
                raise FloatingPointError("Pérdida de ajuste no finita.")
            self.scaler.scale(loss).backward()
            loss_sum += float(summed.detach())
            seen += len(y)
            if (step + 1) % self.accum == 0 or step + 1 == len(loader):
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                previous_scale = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                # Un overflow fp16 omite el paso del optimizador y también el del scheduler.
                if self.scaler.get_scale() >= previous_scale:
                    self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
        self.sync()
        return loss_sum / seen, seen / max(time.perf_counter() - started, 1e-9)

    @torch.no_grad()
    def _validate(self, loader):
        self.network.eval()
        loss_sum, count, ys, probabilities = 0.0, 0, [], []
        for x, y in loader:
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            with self.autocast():
                logits = self.network(x)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="sum")
            loss_sum += float(loss)
            count += len(y)
            ys.append(y.cpu().numpy())
            probabilities.append(logits.float().sigmoid().cpu().numpy())
        y, p = np.concatenate(ys), np.concatenate(probabilities)
        if not np.isfinite(loss_sum) or not np.isfinite(p).all():
            raise FloatingPointError("Pérdida o probabilidades de validación no finitas.")
        auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None
        return loss_sum / count, auc, float(np.mean((p - y) ** 2))

    def _rng_state(self):
        return {"python": random.getstate(), "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(), "loader": self.generator.get_state(),
                "cuda": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None}

    def _save_checkpoint(self, name):
        state = {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                 "scheduler": self.scheduler.state_dict(), "scaler": self.scaler.state_dict(),
                 "epoch": self.epoch, "best_metric": self.best_metric, "bad_epochs": self.bad_epochs,
                 "rng": self._rng_state(), "hyperparams": self.hyperparams, "settings": self.settings,
                 "normalization": self.normalization, "model_name": self.model_name,
                 "fold": self.fold, "signature": self.signature, "history": self.history,
                 "run_info": self.info}
        path = self.out / "checkpoints" / name
        temporary = path.with_suffix(".tmp")
        torch.save(state, temporary)
        os.replace(temporary, path)

    def load_checkpoint(self, path):
        # Solo cargar checkpoints propios: contienen estados Python además de tensores.
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["hyperparams"] != self.hyperparams or state["settings"] != self.settings:
            raise ValueError("Cambió la configuración; use otra salida para un experimento nuevo.")
        if state["model_name"] != self.model_name or state["fold"] != self.fold:
            raise ValueError("Checkpoint de otro modelo/fold.")
        if self.signature is not None and state["signature"] != self.signature:
            raise ValueError("Los datos cambiaron; no se puede reanudar este checkpoint.")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is None:
            raise RuntimeError("Llame prepare antes de cargar el checkpoint.")
        self.scheduler.load_state_dict(state["scheduler"])
        # Al cambiar de GPU, AMP se elige de nuevo; solo fp16 necesita scaler.
        if self.scaler.is_enabled() and state["scaler"]:
            self.scaler.load_state_dict(state["scaler"])
        self.epoch, self.best_metric = state["epoch"], state["best_metric"]
        self.bad_epochs, self.history = state["bad_epochs"], state["history"]
        self.normalization = state["normalization"]
        random.setstate(state["rng"]["python"])
        np.random.set_state(state["rng"]["numpy"])
        torch.set_rng_state(state["rng"]["torch"])
        self.generator.set_state(state["rng"]["loader"])
        cuda_rng = state["rng"]["cuda"]
        if self.device.type == "cuda" and cuda_rng:
            for device_id, rng in enumerate(cuda_rng[:torch.cuda.device_count()]):
                torch.cuda.set_rng_state(rng, device_id)
        return state

    def _write_metrics(self):
        path = self.out / "metrics.csv"
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS)
            writer.writeheader()
            writer.writerows(self.history)
        os.replace(temporary, path)
        if self.aggregate_metrics is not None:
            aggregate = self.aggregate_metrics
            aggregate.parent.mkdir(parents=True, exist_ok=True)
            existing = pd.read_csv(aggregate) if aggregate.exists() else pd.DataFrame(columns=METRIC_COLUMNS)
            existing = existing.loc[existing["fold"].astype(str) != str(self.fold)]
            frames = [existing, pd.DataFrame(self.history, columns=METRIC_COLUMNS)]
            merged = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
            temporary = aggregate.with_suffix(".csv.tmp")
            merged[METRIC_COLUMNS].to_csv(temporary, index=False)
            os.replace(temporary, aggregate)

    def fit(self, train, validation, *, normalization, resume=False):
        if not len(train) or not len(validation):
            raise ValueError("Ajuste y calibración deben contener muestras.")
        self.signature = dataset_signature(train, validation)
        self.normalization = normalization
        self.prepare(len(train))
        checkpoint = self.out / "checkpoints/last.pt"
        if checkpoint.exists() and not resume:
            raise FileExistsError("La salida contiene un checkpoint; use resume=True u otra carpeta.")
        if resume and checkpoint.exists():
            state = self.load_checkpoint(checkpoint)
            if state["normalization"] != normalization:
                raise ValueError("Cambió la normalización del fold.")
            info_path = self.out / "run_info.json"
            previous = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else state["run_info"]
            self.info["started_at"] = previous["started_at"]
            self.info["total_seconds"] = previous["total_seconds"]
            self._write_metrics()
        previous_seconds = self.info["total_seconds"]
        started = time.perf_counter()
        self.info.update(status="running", finished_at=None)
        write_json(self.out / "run_info.json", self.info)
        try:
            train_loader, val_loader = self.loader(train, True), self.loader(validation)
            for epoch in range(self.epoch + 1, self.epochs + 1):
                if self.bad_epochs >= self.patience:
                    break
                self.sync()
                epoch_start = time.perf_counter()
                if self.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(self.device)
                train_loss, throughput = self._train_epoch(train_loader)
                val_loss, auc, brier = self._validate(val_loader)
                self.sync()
                self.epoch = epoch
                improved = val_loss < self.best_metric
                if improved:
                    self.best_metric, self.bad_epochs = val_loss, 0
                else:
                    self.bad_epochs += 1
                cuda = self.device.type == "cuda"
                self.history.append(dict(epoch=epoch, fold=self.fold, train_loss=train_loss,
                    val_loss=val_loss, val_auc=auc, val_brier=brier,
                    lr=self.optimizer.param_groups[0]["lr"], epoch_seconds=time.perf_counter() - epoch_start,
                    samples_per_sec=throughput,
                    gpu_mem_alloc_gb=torch.cuda.memory_allocated(self.device) / 2 ** 30 if cuda else 0.0,
                    gpu_mem_peak_gb=torch.cuda.max_memory_allocated(self.device) / 2 ** 30 if cuda else 0.0))
                self.info["total_seconds"] = previous_seconds + time.perf_counter() - started
                if improved:
                    self._save_checkpoint("best.pt")
                self._save_checkpoint("last.pt")
                self._write_metrics()
                write_json(self.out / "run_info.json", self.info)
                print(f"Fold {self.fold} época {epoch}: train={train_loss:.5f} val={val_loss:.5f}", flush=True)
            best = torch.load(self.out / "checkpoints/best.pt", map_location="cpu", weights_only=False)
            self.model.load_state_dict(best["model"])
            self.info["status"] = "finished"
        except KeyboardInterrupt:
            self.info["status"] = "interrupted"
            raise
        except BaseException:
            self.info["status"] = "failed"
            raise
        finally:
            self.info["total_seconds"] = previous_seconds + time.perf_counter() - started
            self.info["finished_at"] = utc_now()
            write_json(self.out / "run_info.json", self.info)
        return self.history

    @torch.no_grad()
    def predict(self, dataset):
        self.network.eval()
        output = []
        for x, _ in self.loader(dataset):
            with self.autocast():
                logits = self.network(x.to(self.device, non_blocking=True))
            output.append(logits.float().sigmoid().cpu().numpy())
        return np.concatenate(output) if output else np.empty(0, dtype=np.float32)

    def calibrate_predict(self, fold, labels):
        from quant_ai.models.ml import PlattCalibrator

        raw_cal = self.predict(fold.calibration)
        y_cal = fold.calibration.labels[fold.calibration.positions]
        calibrator = PlattCalibrator().fit(raw_cal, y_cal)
        write_json(self.out / "calibrator.json", {"a": calibrator.a, "b": calibrator.b,
                                                  "single_class": bool(len(np.unique(y_cal)) < 2)})
        raw = self.predict(fold.test)
        output = pd.DataFrame({"p_seq": calibrator.transform(raw), "p_seq_raw": raw}, index=fold.test.times)
        output = output.join(labels[["y", "fwd_ret"]], how="left")
        output["symbol"], output["fold"] = fold.symbol, fold.fold.k
        return output.rename_axis("open_time")[["symbol", "fold", "p_seq", "p_seq_raw", "y", "fwd_ret"]]
