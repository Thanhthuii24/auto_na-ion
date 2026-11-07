"""BatteryLife NA-ion data pipeline (paper splits 2021/42/2024; env BL_SPLIT, default 2021).
Preprocessing copied from v4_auto (MIX_large) and BatteryLife data_loader.py.

The implementation mirrors BatteryLife/data_provider/data_loader.py while
keeping final-test access out of ordinary training. Importing this module is
side-effect free; full preprocessing requires the explicit run gate.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import math
import pickle
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
BATTERYLIFE_ROOT = WORKSPACE / "BatteryLife"
DATASET_ROOT = WORKSPACE / "dataset"
LABEL_ROOT = DATASET_ROOT / "Life labels"
DATASET_NAME = "NA-ion"
DATASET_FOLDER = "NA-ion"
LABEL_PREFIX = "NA-ion"
SPLIT_RECORDER_PREFIX = {'2021': 'NAion_2021', '42': 'NAion_42', '2024': 'NAion_2024'}
SPLIT_SEEDS = ("2021", "42", "2024")
SPLIT = os.environ.get("BL_SPLIT", "2021")
if SPLIT not in SPLIT_SEEDS:
    raise ValueError(f"BL_SPLIT must be one of {SPLIT_SEEDS}, got {SPLIT}")
CACHE_TAG = "naion"
DEFAULT_CACHE = ROOT / ".cache" / f"batterylife_{CACHE_TAG}_split{SPLIT}_v1.pt"
# BatteryLife data_loader.py: these ZN-coin cells are charge-first; every other ZN-coin cell is discharge-first.
ZN_COIN_CHARGE_FIRST = {
    "ZN-coin_402-1_20231209225636_01_1.pkl", "ZN-coin_402-2_20231209225727_01_2.pkl",
    "ZN-coin_402-3_20231209225844_01_3.pkl", "ZN-coin_403-1_20231209225922_01_4.pkl",
    "ZN-coin_428-1_20231212185048_01_2.pkl", "ZN-coin_428-2_20231212185058_01_4.pkl",
    "ZN-coin_429-1_20231212185129_01_5.pkl", "ZN-coin_429-2_20231212185157_01_8.pkl",
    "ZN-coin_430-1_20231212185250_02_6.pkl", "ZN-coin_430-2_20231212185305_02_7.pkl",
    "ZN-coin_430-3_20231212185323_03_2.pkl",
}

EARLY_CYCLES = 100
CURVE_LENGTH = 300
HALF_CURVE_LENGTH = CURVE_LENGTH // 2
CHANNELS = 3
NEEDED_KEYS = (
    "current_in_A",
    "voltage_in_V",
    "charge_capacity_in_Ah",
    "discharge_capacity_in_Ah",
    "time_in_s",
)

FOLDER_BY_PREFIX = {
    "MATR": "MATR",
    "HUST": "HUST",
    "SNL": "SNL",
    "CALCE": "CALCE",
    "HNEI": "HNEI",
    "RWTH": "RWTH",
    "UL-PUR": "UL_PUR",
    "Tongji1": "Tongji",
    "Tongji2": "Tongji",
    "Tongji3": "Tongji",
    "Stanford": "Stanford",
    "ISU-ILCC": "ISU_ILCC",
    "XJTU": "XJTU",
}


@dataclass(frozen=True)
class LabelScaler:
    mean: float
    scale: float

    def transform(self, value):
        return (value - self.mean) / self.scale

    def inverse_transform(self, value):
        return value * self.scale + self.mean


@lru_cache(maxsize=1)
def _load_split_recorder():
    path = BATTERYLIFE_ROOT / "data_provider" / "data_split_recorder.py"
    spec = importlib.util.spec_from_file_location("batterylife_split_recorder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import split recorder: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.split_recorder


@lru_cache(maxsize=1)
def load_split_lists(split=None):
    recorder = _load_split_recorder()
    prefix = SPLIT_RECORDER_PREFIX[split or SPLIT]
    return {
        "train": tuple(getattr(recorder, f"{prefix}_train_files")),
        "val": tuple(getattr(recorder, f"{prefix}_val_files")),
        "test": tuple(getattr(recorder, f"{prefix}_test_files")),
    }


@lru_cache(maxsize=1)
def _mich_exp_names():
    recorder = _load_split_recorder()
    return set(
        recorder.MICH_EXP_train_files
        + recorder.MICH_EXP_val_files
        + recorder.MICH_EXP_test_files
    )


def resolve_cell_path(filename):
    if not filename.startswith(DATASET_NAME + "_"):
        raise ValueError(f"Unsupported {DATASET_NAME} filename: {filename}")
    return DATASET_ROOT / DATASET_FOLDER / filename


def _label_path(filename):
    return LABEL_ROOT / f"{LABEL_PREFIX}_labels.json"


@lru_cache(maxsize=None)
def _load_label_file(label_path):
    with Path(label_path).open() as handle:
        return json.load(handle)


def load_eol(filename):
    labels = _load_label_file(_label_path(filename))
    key = filename.replace("--", "-#") if filename.startswith("Tongji") else filename
    value = labels.get(key)
    return None if value is None else float(value)


def audit_inventory():
    splits = load_split_lists()
    sets = {name: set(values) for name, values in splits.items()}
    overlaps = sorted(
        (sets["train"] & sets["val"])
        | (sets["train"] & sets["test"])
        | (sets["val"] & sets["test"])
    )
    report = {
        "listed_cells": {name: len(values) for name, values in splits.items()},
        "cross_split_filename_overlap": overlaps,
        "missing_files": [],
        "missing_labels": [],
        "eol_not_after_100": [],
    }
    for split, filenames in splits.items():
        for filename in filenames:
            path = resolve_cell_path(filename)
            if not path.exists():
                report["missing_files"].append({"split": split, "filename": filename})
                continue
            eol = load_eol(filename)
            if eol is None:
                report["missing_labels"].append({"split": split, "filename": filename})
            elif eol <= EARLY_CYCLES:
                report["eol_not_after_100"].append(
                    {"split": split, "filename": filename, "eol": eol}
                )
    return report


def nominal_capacity(filename, cell):
    if filename.startswith("RWTH"):
        value = 1.85
    elif filename.startswith("SNL_18650_NCA_25C_20-80"):
        value = 3.2
    else:
        value = float(cell["nominal_capacity_in_Ah"])
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid nominal capacity for {filename}: {value}")
    return value


def reference_resample(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Cannot resample an empty or non-finite curve")
    raw_bases = np.arange(1, len(values) + 1)
    interp_bases = np.linspace(1, len(values) + 1, num=HALF_CURVE_LENGTH, endpoint=True)
    return np.interp(interp_bases, raw_bases, values)


def _cycle_frame(cycle, cycle_number):
    frame = pd.DataFrame({key: cycle[key] for key in NEEDED_KEYS})
    frame["cycle_number"] = cycle_number
    frame.loc[frame["charge_capacity_in_Ah"] < 0] = np.nan
    frame.loc[frame["discharge_capacity_in_Ah"] < 0] = np.nan
    frame.bfill(inplace=True)
    return frame


def preprocess_cycle(frame, filename, capacity):
    voltage = frame["voltage_in_V"].to_numpy(dtype=np.float64)
    current = frame["current_in_A"].to_numpy(dtype=np.float64)
    charge_q = frame["charge_capacity_in_Ah"].to_numpy(dtype=np.float64)
    discharge_q = frame["discharge_capacity_in_Ah"].to_numpy(dtype=np.float64)
    current_c = current / capacity

    charge_indices = np.flatnonzero(current_c >= 0.01)
    discharge_indices = np.flatnonzero(current_c <= -0.01)
    if not len(charge_indices) or not len(discharge_indices):
        raise ValueError(f"Missing charge/discharge segment in {filename}")
    charge_end = int(charge_indices[-1])
    discharge_end = int(discharge_indices[-1])

    if filename.startswith("RWTH") or (filename.startswith("ZN-coin") and filename not in ZN_COIN_CHARGE_FIRST):
        discharge = (voltage[:discharge_end], current[:discharge_end], discharge_q[:discharge_end])
        charge = (voltage[discharge_end:], current[discharge_end:], charge_q[discharge_end:])
        keep = np.abs(charge[1] / capacity) > 0.01
        charge = tuple(values[keep] for values in charge)
    else:
        discharge = (voltage[charge_end:], current[charge_end:], discharge_q[charge_end:])
        keep = np.abs(discharge[1] / capacity) > 0.01
        discharge = tuple(values[keep] for values in discharge)
        charge = (voltage[:charge_end], current[:charge_end], charge_q[:charge_end])

    charge = tuple(reference_resample(values) for values in charge)
    discharge = tuple(reference_resample(values) for values in discharge)
    voltage_curve = np.concatenate((charge[0], discharge[0]))
    current_curve = np.concatenate((charge[1], discharge[1]))
    capacity_curve = np.concatenate((charge[2], discharge[2]))
    voltage_max = float(np.max(voltage_curve))
    if not math.isfinite(voltage_max) or voltage_max == 0:
        raise ValueError(f"Invalid voltage normalization for {filename}")
    curve = np.stack(
        (voltage_curve / voltage_max, current_curve / capacity, capacity_curve / capacity)
    ).astype(np.float32)
    if curve.shape != (CHANNELS, CURVE_LENGTH) or not np.isfinite(curve).all():
        raise ValueError(f"Invalid processed curve for {filename}: {curve.shape}")
    return curve


def preprocess_cell(filename, split):
    path = resolve_cell_path(filename)
    if not path.exists():
        return {"status": "excluded", "reason": "missing_file", "filename": filename, "split": split}
    eol = load_eol(filename)
    if eol is None:
        return {"status": "excluded", "reason": "missing_official_eol", "filename": filename, "split": split}
    if eol <= EARLY_CYCLES:
        return {"status": "excluded", "reason": "eol_not_after_100", "filename": filename, "split": split, "eol": eol}

    try:
        with path.open("rb") as handle:
            cell = pickle.load(handle)
        cycles = cell["cycle_data"]
        capacity = nominal_capacity(filename, cell)
        tensor = np.zeros((EARLY_CYCLES, CHANNELS, CURVE_LENGTH), dtype=np.float32)
        observed = min(EARLY_CYCLES, len(cycles))
        for index, cycle in enumerate(cycles[:observed], start=1):
            tensor[index - 1] = preprocess_cycle(_cycle_frame(cycle, index), filename, capacity)
        if not np.isfinite(tensor).all():
            raise ValueError("non-finite cell tensor")
        return {
            "status": "accepted",
            "filename": filename,
            "split": split,
            "domain": DATASET_NAME,
            "eol": eol,
            "recorded_cycles": len(cycles),
            "observed_cycles": observed,
            "tensor": torch.from_numpy(tensor),
        }
    except Exception as exc:
        return {
            "status": "rejected",
            "reason": f"{type(exc).__name__}: {str(exc)[:200]}",
            "filename": filename,
            "split": split,
        }


def fit_label_scaler(train_cells):
    values = np.asarray([cell["eol"] for cell in train_cells], dtype=np.float64)
    if values.size == 0:
        raise ValueError("No training labels")
    scale = float(values.std(ddof=0))
    if scale <= 0 or not math.isfinite(scale):
        raise ValueError("Invalid training-label scale")
    return LabelScaler(mean=float(values.mean()), scale=scale)


class BatteryPrefixDataset(Dataset):
    def __init__(self, cells, scaler):
        self.cells = cells
        self.scaler = scaler
        self.index = []
        for cell_index, cell in enumerate(cells):
            maximum = min(EARLY_CYCLES, int(cell["observed_cycles"]), math.ceil(cell["eol"]) - 1)
            self.index.extend((cell_index, prefix) for prefix in range(1, maximum + 1))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        cell_index, prefix = self.index[item]
        cell = self.cells[cell_index]
        mask = torch.zeros(EARLY_CYCLES, dtype=torch.float32)
        mask[:prefix] = 1
        curves = cell["tensor"].clone() * mask[:, None, None]
        return {
            "cycle_curve_data": curves,
            "curve_attn_mask": mask,
            "label": torch.tensor(self.scaler.transform(cell["eol"]), dtype=torch.float32),
            "raw_eol": torch.tensor(cell["eol"], dtype=torch.float32),
            "cell_id": cell["filename"],
            "domain": cell["domain"],
            "prefix_length": prefix,
        }


def compute_metrics(predictions, targets):
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    if predictions.shape != targets.shape or not len(targets) or np.any(targets == 0):
        raise ValueError("Invalid metric inputs")
    relative = np.abs((predictions - targets) / targets)
    return {
        "mape": float(relative.mean()),
        "acc15": float((relative <= 0.15).mean()),
        "mae": float(np.abs(predictions - targets).mean()),
        "rmse": float(np.sqrt(np.square(predictions - targets).mean())),
    }


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_cache_build_authority():
    with (ROOT / "state.json").open() as handle:
        state = json.load(handle)
    with (ROOT / "protocol_lock.json").open() as handle:
        lock = json.load(handle)
    if not state.get("run_approved"):
        raise RuntimeError("Run gate is closed; full cache construction is forbidden")
    if lock.get("status") not in {"approved", "locked"}:
        raise RuntimeError("Protocol is not approved")


def build_cache(output_path=DEFAULT_CACHE):
    require_cache_build_authority()
    started = time.perf_counter()
    split_lists = load_split_lists()
    accepted = {split: [] for split in split_lists}
    audit = {
        "inventory": audit_inventory(),
        "accepted": {},
        "excluded": [],
        "rejected": [],
    }
    for split, filenames in split_lists.items():
        for index, filename in enumerate(filenames, start=1):
            result = preprocess_cell(filename, split)
            if result["status"] == "accepted":
                accepted[split].append(result)
            else:
                audit[result["status"]].append(result)
            if index % 25 == 0 or index == len(filenames):
                print(f"[{split}] processed {index}/{len(filenames)} cells", flush=True)
        audit["accepted"][split] = len(accepted[split])
    scaler = fit_label_scaler(accepted["train"])
    payload = {
        "protocol_version": f"BATTERYLIFE-{DATASET_NAME.upper()}-SPLIT{SPLIT}-V1",
        "split_seed": SPLIT,
        "cells": accepted,
        "label_scaler": {"mean": scaler.mean, "scale": scaler.scale},
        "audit": audit,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    audit["cache_build_wall_seconds"] = round(time.perf_counter() - started, 3)
    audit_path = output_path.with_suffix(".audit.json")
    with audit_path.open("w") as handle:
        json.dump(audit, handle, indent=2)
        handle.write("\n")
    return output_path, sha256(output_path), audit_path, audit


def _load_locked_cache(cache_path=None):
    with (ROOT / "protocol_lock.json").open() as handle:
        lock = json.load(handle)
    if lock.get("status") != "locked":
        raise RuntimeError("Protocol/cache is not locked")
    split_entry = lock.get("split_caches", {}).get(SPLIT) if cache_path is None else None
    if split_entry:
        path = Path(split_entry["path"])
        expected = split_entry["sha256"]
    else:
        path = Path(cache_path or lock["cache_path"])
        expected = lock.get("cache_sha256")
    if not path.is_absolute():
        path = ROOT / path
    if cache_path is not None and Path(cache_path).name == Path(lock.get("cache_v2_path", "")).name:
        expected = lock.get("cache_v2_sha256")  # amendment A6 derived cache
    if cache_path is not None and Path(cache_path).name == Path(lock.get("cache_v3_path", "")).name:
        expected = lock.get("cache_v3_sha256")  # amendment A7 derived cache
    if sha256(path) != expected:
        raise RuntimeError("Cache SHA-256 mismatch")
    return torch.load(path, weights_only=False)


def load_train_val_data(batch_size, shuffle_seed, cache_path=None):
    payload = _load_locked_cache(cache_path)
    scaler = LabelScaler(**payload["label_scaler"])
    generator = torch.Generator().manual_seed(shuffle_seed)
    train = BatteryPrefixDataset(payload["cells"]["train"], scaler)
    val = BatteryPrefixDataset(payload["cells"]["val"], scaler)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True, generator=generator),
        DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=False),
        scaler,
    )


def load_final_test_data(batch_size, cache_path=None):
    with (ROOT / "state.json").open() as handle:
        state = json.load(handle)
    if state.get("phase") != "final_evaluation" or state.get("test_status") != "authorized":
        raise RuntimeError("Final-test access is locked")
    payload = _load_locked_cache(cache_path)
    scaler = LabelScaler(**payload["label_scaler"])
    test = BatteryPrefixDataset(payload["cells"]["test"], scaler)
    return DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=False), scaler


if __name__ == "__main__":
    raise SystemExit(
        "Use build_cache() only after the explicit run gate; importing prepare.py is safe."
    )
