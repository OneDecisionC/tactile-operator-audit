r"""
Braille temporal ablations and non-SNN baselines.

Primary protocol mirrors the existing Braille Pareto runs:
  th1 spike input, ON/OFF polarity separated, 25 ms bins, crop_start=7,
  stratified 5-fold CV, fixed final epoch as the main metric.

Examples:
  python run_braille_ablation_baselines.py --models snn --variants original reverse shuffle count
  python run_braille_ablation_baselines.py --models mlp conv1d gru --variants original count
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import snntorch as snn


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stratified_kfold(y_np: np.ndarray, n_splits: int = 5, seed: int = 42):
    rng = np.random.RandomState(seed)
    class_indices = defaultdict(list)
    for i, label in enumerate(y_np):
        class_indices[int(label)].append(i)

    folds = [[] for _ in range(n_splits)]
    for indices in class_indices.values():
        rng.shuffle(indices)
        n_per_fold = len(indices) // n_splits
        for fold in range(n_splits):
            start = fold * n_per_fold
            end = start + n_per_fold if fold < n_splits - 1 else len(indices)
            folds[fold].extend(indices[start:end])

    for fold in range(n_splits):
        rng.shuffle(folds[fold])

    for fold in range(n_splits):
        val_idx = np.array(folds[fold], dtype=np.int64)
        train_idx = np.concatenate(
            [np.array(folds[i], dtype=np.int64) for i in range(n_splits) if i != fold]
        )
        yield train_idx, val_idx


def load_braille_events(
    threshold: str,
    event_dt: float,
    event_encoding: str,
    crop_start: int,
    crop_end: int,
):
    data_path = Path("braille_letters_dataset") / "data" / f"data_braille_letters_{threshold}"
    with data_path.open("rb") as f:
        samples = pickle.load(f, encoding="latin1")

    letters = sorted({sample["letter"] for sample in samples})
    label_to_idx = {letter: i for i, letter in enumerate(letters)}
    duration = 1.3
    time_steps = int(np.ceil(duration / event_dt))
    n_channels = 24 if event_encoding == "polarity_binary" else 12
    x = np.zeros((len(samples), time_steps, n_channels), dtype=np.float32)
    y = np.zeros(len(samples), dtype=np.int64)

    raw_events = 0
    for sample_index, sample in enumerate(samples):
        for taxel_index, (on_times, off_times) in enumerate(sample["events"]):
            raw_events += len(on_times) + len(off_times)
            for t in on_times:
                step = min(int(t / event_dt), time_steps - 1)
                x[sample_index, step, taxel_index] = 1.0
            for t in off_times:
                step = min(int(t / event_dt), time_steps - 1)
                channel = taxel_index + 12 if event_encoding == "polarity_binary" else taxel_index
                x[sample_index, step, channel] = 1.0
        y[sample_index] = label_to_idx[sample["letter"]]

    crop_stop = x.shape[1] if crop_end < 0 else min(crop_end, x.shape[1])
    if crop_start >= crop_stop:
        raise ValueError("crop_start must be smaller than crop_end")
    if crop_start > 0 or crop_stop < x.shape[1]:
        x = x[:, crop_start:crop_stop, :]

    meta = {
        "classes": len(letters),
        "samples": len(samples),
        "raw_events": raw_events,
        "retained_events": float(x.sum()),
        "active_inputs_per_sample": float(x.sum() / len(samples)),
        "input_density": float(x.mean()),
        "time_steps": int(x.shape[1]),
        "channels": int(x.shape[2]),
        "letters": letters,
    }
    return x, y, meta


def transform_variant(x: np.ndarray, variant: str, seed: int) -> np.ndarray:
    if variant == "original":
        return x.copy()
    if variant == "reverse":
        return x[:, ::-1, :].copy()
    if variant == "shuffle":
        shuffled = np.empty_like(x)
        for i in range(x.shape[0]):
            rng = np.random.default_rng(seed + i)
            shuffled[i] = x[i, rng.permutation(x.shape[1]), :]
        return shuffled
    if variant == "count":
        return x.sum(axis=1, keepdims=True).astype(np.float32)
    raise ValueError(f"Unknown variant: {variant}")


class RateSNNClassifier(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_hidden: int,
        n_output: int,
        beta: float,
        threshold: float,
        dropout: float,
        layers: int,
        readout: str,
    ):
        super().__init__()
        self.readout = readout
        self.fc_layers = nn.ModuleList()
        self.lif_layers = nn.ModuleList()
        self.drop_layers = nn.ModuleList()
        in_dim = n_input
        for _ in range(layers - 1):
            self.fc_layers.append(nn.Linear(in_dim, n_hidden))
            self.lif_layers.append(snn.Leaky(beta=beta, threshold=threshold))
            self.drop_layers.append(nn.Dropout(dropout))
            in_dim = n_hidden
        self.fc_out = nn.Linear(n_hidden, n_output)
        self.lif_out = snn.Leaky(beta=beta, threshold=threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, _ = x.shape
        mems = [lif.init_leaky() for lif in self.lif_layers]
        mem_out = self.lif_out.init_leaky()
        spk_sum = torch.zeros(batch, self.fc_out.out_features, device=x.device)
        mem_sum = torch.zeros(batch, self.fc_out.out_features, device=x.device)
        for t in range(steps):
            cur = x[:, t, :]
            for i, (fc, lif, drop) in enumerate(
                zip(self.fc_layers, self.lif_layers, self.drop_layers)
            ):
                cur = fc(cur)
                cur, mems[i] = lif(drop(cur), mems[i])
            cur = self.fc_out(cur)
            spk, mem_out = self.lif_out(cur, mem_out)
            spk_sum += spk
            mem_sum += mem_out
        if self.readout == "spikes":
            return spk_sum
        if self.readout == "membrane_sum":
            return mem_sum
        if self.readout == "membrane_final":
            return mem_out
        raise ValueError(f"Unknown SNN readout: {self.readout}")


class RecurrentSNNClassifier(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_hidden: int,
        n_output: int,
        beta: float,
        threshold: float,
        dropout: float,
        readout: str,
    ):
        super().__init__()
        self.readout = readout
        self.fc_in = nn.Linear(n_input, n_hidden)
        self.fc_rec = nn.Linear(n_hidden, n_hidden, bias=False)
        self.lif_hidden = snn.Leaky(beta=beta, threshold=threshold)
        self.drop = nn.Dropout(dropout)
        self.fc_out = nn.Linear(n_hidden, n_output)
        self.lif_out = snn.Leaky(beta=beta, threshold=threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, _ = x.shape
        mem_hidden = self.lif_hidden.init_leaky()
        mem_out = self.lif_out.init_leaky()
        spk_hidden = torch.zeros(batch, self.fc_rec.in_features, device=x.device)
        spk_sum = torch.zeros(batch, self.fc_out.out_features, device=x.device)
        mem_sum = torch.zeros(batch, self.fc_out.out_features, device=x.device)
        for t in range(steps):
            cur_hidden = self.fc_in(x[:, t, :]) + self.fc_rec(spk_hidden)
            spk_hidden, mem_hidden = self.lif_hidden(self.drop(cur_hidden), mem_hidden)
            cur_out = self.fc_out(spk_hidden)
            spk_out, mem_out = self.lif_out(cur_out, mem_out)
            spk_sum += spk_out
            mem_sum += mem_out
        if self.readout == "spikes":
            return spk_sum
        if self.readout == "membrane_sum":
            return mem_sum
        if self.readout == "membrane_final":
            return mem_out
        raise ValueError(f"Unknown SNN readout: {self.readout}")


class MLPClassifier(nn.Module):
    def __init__(self, n_input: int, n_hidden: int, n_output: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_hidden, n_output),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(1))


class Conv1DClassifier(nn.Module):
    def __init__(self, n_channels: int, n_hidden: int, n_output: int, dropout: float):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, n_hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(n_hidden, n_hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(n_hidden, n_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x.transpose(1, 2)).squeeze(-1)
        return self.head(z)


class CausalChomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            CausalChomp1d(padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            CausalChomp1d(padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.out = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.net(x) + self.downsample(x))


class TCNClassifier(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_hidden: int,
        n_output: int,
        dropout: float,
        n_blocks: int,
        kernel_size: int = 3,
    ):
        super().__init__()
        blocks = []
        in_channels = n_channels
        for block_index in range(n_blocks):
            blocks.append(
                TCNBlock(
                    in_channels=in_channels,
                    out_channels=n_hidden,
                    kernel_size=kernel_size,
                    dilation=2 ** block_index,
                    dropout=dropout,
                )
            )
            in_channels = n_hidden
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(n_hidden, n_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x.transpose(1, 2))
        z = self.pool(z).squeeze(-1)
        return self.head(z)


class GRUClassifier(nn.Module):
    def __init__(self, n_input: int, n_hidden: int, n_output: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_input,
            hidden_size=n_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(n_hidden, n_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)
        return self.head(self.drop(h[-1]))


def make_model(model_name: str, x_shape, n_classes: int, args) -> nn.Module:
    _, time_steps, channels = x_shape
    if model_name == "snn":
        return RateSNNClassifier(
            n_input=channels,
            n_hidden=args.hidden,
            n_output=n_classes,
            beta=args.beta,
            threshold=args.threshold_v,
            dropout=args.dropout,
            layers=args.layers,
            readout=getattr(args, "snn_readout", "spikes"),
        )
    if model_name == "rsnn":
        return RecurrentSNNClassifier(
            n_input=channels,
            n_hidden=args.hidden,
            n_output=n_classes,
            beta=args.beta,
            threshold=args.threshold_v,
            dropout=args.dropout,
            readout=getattr(args, "snn_readout", "spikes"),
        )
    if model_name == "mlp":
        return MLPClassifier(time_steps * channels, args.hidden, n_classes, args.dropout)
    if model_name == "conv1d":
        return Conv1DClassifier(channels, args.hidden, n_classes, args.dropout)
    if model_name == "tcn":
        return TCNClassifier(channels, args.hidden, n_classes, args.dropout, args.layers)
    if model_name == "gru":
        return GRUClassifier(channels, args.hidden, n_classes, args.dropout)
    raise ValueError(f"Unknown model: {model_name}")


def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * len(yb)
        correct += int((logits.argmax(1) == yb).sum().item())
        total += int(len(yb))
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        total_loss += float(loss.item()) * len(yb)
        correct += int((logits.argmax(1) == yb).sum().item())
        total += int(len(yb))
    return total_loss / total, correct / total


def run_cv(x_np: np.ndarray, y_np: np.ndarray, model_name: str, variant: str, args, device):
    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(
        stratified_kfold(y_np, n_splits=args.folds, seed=args.seed)
    ):
        fold_seed = args.seed + fold
        seed_everything(fold_seed)
        x_train = torch.tensor(x_np[train_idx], dtype=torch.float32)
        y_train = torch.tensor(y_np[train_idx], dtype=torch.long)
        x_val = torch.tensor(x_np[val_idx], dtype=torch.float32)
        y_val = torch.tensor(y_np[val_idx], dtype=torch.long)
        train_loader = DataLoader(
            TensorDataset(x_train, y_train),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
        val_loader = DataLoader(
            TensorDataset(x_val, y_val),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
        model = make_model(model_name, x_np.shape, int(y_np.max()) + 1, args).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        loss_fn = nn.CrossEntropyLoss()
        best_acc = 0.0
        final_acc = 0.0
        final_loss = 0.0
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, loss_fn, device)
            val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
            scheduler.step()
            best_acc = max(best_acc, val_acc)
            final_acc = val_acc
            final_loss = val_loss
            if args.log_every and (epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs):
                print(
                    f"{model_name}/{variant} fold {fold + 1}/{args.folds} "
                    f"epoch {epoch:03d}: train={train_acc * 100:.1f}% "
                    f"val={val_acc * 100:.1f}% best={best_acc * 100:.1f}%"
                )
        fold_results.append(
            {
                "fold": fold + 1,
                "seed": fold_seed,
                "final_acc": final_acc,
                "best_acc": best_acc,
                "final_loss": final_loss,
                "parameters": sum(p.numel() for p in model.parameters()),
            }
        )
    final_accs = np.array([r["final_acc"] for r in fold_results], dtype=np.float64)
    best_accs = np.array([r["best_acc"] for r in fold_results], dtype=np.float64)
    return {
        "model": model_name,
        "variant": variant,
        "folds": fold_results,
        "mean_final_acc": float(final_accs.mean()),
        "std_final_acc": float(final_accs.std()),
        "mean_best_acc": float(best_accs.mean()),
        "std_best_acc": float(best_accs.std()),
        "parameters": int(fold_results[0]["parameters"]),
    }


def write_outputs(results, meta, args):
    rows = []
    for result in results:
        rows.append(
            {
                "model": result["model"],
                "variant": result["variant"],
                "threshold": args.threshold,
                "event_encoding": args.event_encoding,
                "event_dt_ms": args.event_dt * 1000,
                "crop_start": args.crop_start,
                "crop_end": args.crop_end,
                "time_steps": meta["time_steps"] if result["variant"] != "count" else 1,
                "channels": meta["channels"],
                "hidden": args.hidden,
                "layers": args.layers if result["model"] in ("snn", "rsnn") else "",
                "snn_readout": args.snn_readout if result["model"] in ("snn", "rsnn") else "",
                "parameters": result["parameters"],
                "epochs": args.epochs,
                "folds": args.folds,
                "mean_final_acc": result["mean_final_acc"],
                "std_final_acc": result["std_final_acc"],
                "mean_best_acc": result["mean_best_acc"],
                "std_best_acc": result["std_best_acc"],
                "fold_final_accs": "|".join(f"{fold['final_acc']:.6f}" for fold in result["folds"]),
                "fold_best_accs": "|".join(f"{fold['best_acc']:.6f}" for fold in result["folds"]),
            }
        )

    csv_path = Path(args.csv)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = Path(args.json)
    payload = {"meta": meta, "args": vars(args), "results": results}
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return csv_path, json_path


def parse_args():
    parser = argparse.ArgumentParser(description="Run Braille temporal ablations and baselines.")
    parser.add_argument("--threshold", default="th1", choices=["th1", "th2", "th5", "th10"])
    parser.add_argument("--event_dt", type=float, default=0.025)
    parser.add_argument("--event_encoding", default="polarity_binary", choices=["polarity_binary", "merged_binary"])
    parser.add_argument("--crop_start", type=int, default=7)
    parser.add_argument("--crop_end", type=int, default=-1)
    parser.add_argument("--variants", nargs="+", default=["original", "reverse", "shuffle", "count"])
    parser.add_argument("--models", nargs="+", default=["snn", "mlp", "conv1d", "gru"])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--beta", type=float, default=0.8)
    parser.add_argument("--threshold_v", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--snn_readout", choices=["spikes", "membrane_sum", "membrane_final"], default="spikes")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--csv", default="braille_ablation_baseline_results.csv")
    parser.add_argument("--json", default="braille_ablation_baseline_results.json")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    x_base, y, meta = load_braille_events(
        args.threshold,
        args.event_dt,
        args.event_encoding,
        args.crop_start,
        args.crop_end,
    )
    print(
        f"Loaded Braille: samples={meta['samples']} classes={meta['classes']} "
        f"shape={x_base.shape} active/sample={meta['active_inputs_per_sample']:.1f}"
    )

    results = []
    for variant in args.variants:
        x_variant = transform_variant(x_base, variant, args.seed)
        print(f"\nVariant={variant} shape={x_variant.shape}")
        for model_name in args.models:
            print(f"Running model={model_name}, variant={variant}")
            result = run_cv(x_variant, y, model_name, variant, args, device)
            print(
                f"Done {model_name}/{variant}: final "
                f"{result['mean_final_acc'] * 100:.1f}% +/- {result['std_final_acc'] * 100:.1f}% "
                f"| best diagnostic {result['mean_best_acc'] * 100:.1f}%"
            )
            results.append(result)

    csv_path, json_path = write_outputs(results, meta, args)
    print(f"\nWrote {csv_path.resolve()}")
    print(f"Wrote {json_path.resolve()}")


if __name__ == "__main__":
    main()
