from __future__ import annotations

import argparse
import json
import math
import os
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
WORK_DIR = SCRIPT_DIR / "work"
os.environ.setdefault("MPLCONFIGDIR", str(WORK_DIR / "mplconfig"))
os.environ.setdefault("TORCH_HOME", str(WORK_DIR / "torch_cache"))
WORK_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib
import numpy as np
from scipy.spatial.transform import Rotation
from sklearn.metrics import confusion_matrix

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from LaueTools import lauecore as LAUE


#MATERIALS = ["Si", "Cu", "Ge", "Al", "Fe", "Ni", "Ti", "W", "Au", "Ag"]
MATERIALS = ["Si"]
PYTHONCALIB = [77.088, 1012.45, 1049.92, 0.423, 0.172]
PIXELSIZE = 0.079142
DETECTOR_DIAMETER = PIXELSIZE * 2048
DETECTOR_DIM = (2048, 2048)
ENERGY_BANDPASS_KEV = (5.0, 25.0)
KF_DIRECTION = "Z>0"
OUTPUT_SIZE = 192
SPOT_RADIUS = 2


@dataclass(frozen=True)
class DetectorConfig:
    pythoncalib: Sequence[float] = tuple(PYTHONCALIB)
    pixelsize: float = PIXELSIZE
    detectordiameter: float = DETECTOR_DIAMETER
    dim: Sequence[int] = tuple(DETECTOR_DIM)
    emin: float = ENERGY_BANDPASS_KEV[0]
    emax: float = ENERGY_BANDPASS_KEV[1]
    kf_direction: str = KF_DIRECTION
    output_size: int = OUTPUT_SIZE
    spot_radius: int = SPOT_RADIUS


def simulate_pattern(
    material: str,
    orientation_matrix: np.ndarray,
    detector: DetectorConfig = DetectorConfig(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grain = [None, None, np.asarray(orientation_matrix, dtype=float), material]
    _, _, _, posx, posy, energy = LAUE.SimulateLaue(
        grain,
        detector.emin,
        detector.emax,
        detector.pythoncalib,
        kf_direction=detector.kf_direction,
        removeharmonics=1,
        pixelsize=detector.pixelsize,
        dim=tuple(detector.dim),
        detectordiameter=detector.detectordiameter,
    )

    return np.asarray(posx, dtype=np.float32), np.asarray(posy, dtype=np.float32), np.asarray(
        energy, dtype=np.float32
    )


def render_pattern(
    posx: Iterable[float],
    posy: Iterable[float],
    energy: Iterable[float],
    detector_dim: Sequence[int] = DETECTOR_DIM,
    output_size: int = OUTPUT_SIZE,
    spot_radius: int = SPOT_RADIUS,
) -> np.ndarray:
    image = np.zeros((output_size, output_size), dtype=np.float32)
    posx = np.asarray(posx, dtype=np.float32)
    posy = np.asarray(posy, dtype=np.float32)
    energy = np.asarray(energy, dtype=np.float32)

    if not (len(posx) == len(posy) == len(energy)):
        raise ValueError("posx, posy, and energy must have the same length")

    scale_x = output_size / float(detector_dim[0])
    scale_y = output_size / float(detector_dim[1])

    for x, y, e in zip(posx, posy, energy):
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(e)):
            continue
        sx = int(round(x * scale_x))
        sy = int(round(y * scale_y))
        if sx < 0 or sy < 0 or sx >= output_size or sy >= output_size:
            continue

        x0 = max(0, sx - spot_radius)
        x1 = min(output_size, sx + spot_radius + 1)
        y0 = max(0, sy - spot_radius)
        y1 = min(output_size, sy + spot_radius + 1)

        for yy in range(y0, y1):
            for xx in range(x0, x1):
                if (xx - sx) ** 2 + (yy - sy) ** 2 <= spot_radius**2:
                    image[yy, xx] = max(image[yy, xx], e)

    return image


def generate_dataset(
    output_dir: Path,
    samples_per_material: int = 500,
    materials: Sequence[str] = MATERIALS,
    detector: DetectorConfig = DetectorConfig(),
    seed: int = 0,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    failures_path = output_dir / "failed_simulations.log"
    rng = np.random.default_rng(seed)
    failures: list[str] = []

    for material in materials:
        material_dir = output_dir / material
        material_dir.mkdir(parents=True, exist_ok=True)

        for index in range(samples_per_material):
            try:
                rotation = Rotation.random(random_state=rng).as_matrix()
                posx, posy, energy = simulate_pattern(material, rotation, detector=detector)
                image = render_pattern(
                    posx,
                    posy,
                    energy,
                    detector_dim=detector.dim,
                    output_size=detector.output_size,
                    spot_radius=detector.spot_radius,
                )
                np.save(material_dir / f"{material}_{index:04d}.npy", image)
            except Exception as exc:  # pragma: no cover - logging path
                failures.append(
                    f"{material}_{index:04d}: {type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                )

    failures_path.write_text("\n\n".join(failures), encoding="utf-8")
    print(f"Finished dataset generation in {output_dir}")
    print(f"Failed simulations: {len(failures)}")


def _load_torch():
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, random_split
    from torchvision import models

    return torch, nn, Dataset, DataLoader, random_split, models


class NpyPatternDataset:
    def __init__(self, root_dir: Path, max_energy_kev: float = ENERGY_BANDPASS_KEV[1]):
        self.root_dir = Path(root_dir)
        self.class_names = sorted([path.name for path in self.root_dir.iterdir() if path.is_dir()])
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.samples: list[tuple[Path, int]] = []
        self.max_energy_kev = float(max_energy_kev)

        for class_name in self.class_names:
            for npy_path in sorted((self.root_dir / class_name).glob("*.npy")):
                self.samples.append((npy_path, self.class_to_idx[class_name]))

        if not self.samples:
            raise ValueError(f"No .npy files found under {self.root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        torch, _, _, _, _, _ = _load_torch()
        path, label = self.samples[index]
        image = np.load(path).astype(np.float32)
        image = np.clip(image / self.max_energy_kev, 0.0, 1.0)
        image = np.repeat(image[None, :, :], 3, axis=0)
        tensor = torch.from_numpy(image)
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor, label


def _build_dataloaders(data_dir: Path, batch_size: int, seed: int):
    torch, _, Dataset, DataLoader, random_split, _ = _load_torch()

    class WrappedDataset(Dataset):
        def __init__(self, base: NpyPatternDataset):
            self.base = base
            self.class_names = base.class_names

        def __len__(self):
            return len(self.base)

        def __getitem__(self, index):
            return self.base[index]

    dataset = WrappedDataset(NpyPatternDataset(data_dir))
    total = len(dataset)
    train_len = int(total * 0.8)
    val_len = int(total * 0.1)
    test_len = total - train_len - val_len
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        dataset, [train_len, val_len, test_len], generator=generator
    )

    loaders = {
        "train": DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
        "val": DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0),
        "test": DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0),
    }
    return dataset.class_names, loaders


def train_classifier(
    data_dir: Path,
    model_path: Path,
    outputs_dir: Path,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    seed: int = 0,
) -> dict:
    torch, nn, _, _, _, models = _load_torch()
    class_names, loaders = _build_dataloaders(data_dir, batch_size=batch_size, seed=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    except Exception as exc:
        raise RuntimeError(
            "Unable to load pretrained ResNet18 weights. Ensure torch/torchvision are installed "
            "and pretrained weights are available."
        ) from exc

    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_state = None
    best_val_acc = -1.0

    for epoch in range(epochs):
        for split in ("train", "val"):
            model.train(split == "train")
            running_loss = 0.0
            running_correct = 0
            total = 0

            for inputs, labels in loaders[split]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(split == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    if split == "train":
                        loss.backward()
                        optimizer.step()

                preds = outputs.argmax(dim=1)
                running_loss += loss.item() * inputs.size(0)
                running_correct += (preds == labels).sum().item()
                total += inputs.size(0)

            epoch_loss = running_loss / max(total, 1)
            epoch_acc = running_correct / max(total, 1)
            history[f"{split}_loss"].append(epoch_loss)
            history[f"{split}_acc"].append(epoch_acc)

            if split == "val" and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"train loss {history['train_loss'][-1]:.4f} acc {history['train_acc'][-1]:.4f} | "
            f"val loss {history['val_loss'][-1]:.4f} acc {history['val_acc'][-1]:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    outputs_dir.mkdir(parents=True, exist_ok=True)
    _plot_training_curves(history, outputs_dir / "training_curves.png")

    test_metrics = _evaluate_model(model, loaders["test"], class_names, device)
    _plot_confusion_matrix(
        test_metrics["confusion_matrix"], class_names, outputs_dir / "confusion_matrix.png"
    )

    checkpoint = {
        "state_dict": model.state_dict(),
        "class_names": class_names,
        "detector_dim": tuple(DETECTOR_DIM),
        "output_size": OUTPUT_SIZE,
        "spot_radius": SPOT_RADIUS,
        "max_energy_kev": ENERGY_BANDPASS_KEV[1],
    }
    torch.save(checkpoint, model_path)

    metrics_path = outputs_dir / "metrics.json"
    metrics_payload = {
        "test_accuracy": test_metrics["accuracy"],
        "class_names": class_names,
        "confusion_matrix": test_metrics["confusion_matrix"].tolist(),
        "test_counts": dict(Counter(test_metrics["targets"])),
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print("Confusion matrix:")
    print(test_metrics["confusion_matrix"])

    return metrics_payload


def _evaluate_model(model, dataloader, class_names, device):
    torch, _, _, _, _, _ = _load_torch()
    model.eval()
    all_preds: list[int] = []
    all_targets: list[int] = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs.to(device))
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(labels.numpy().tolist())

    cm = confusion_matrix(all_targets, all_preds, labels=list(range(len(class_names))))
    accuracy = float((np.array(all_preds) == np.array(all_targets)).mean())
    return {"accuracy": accuracy, "confusion_matrix": cm, "predictions": all_preds, "targets": all_targets}


def _plot_training_curves(history: dict, output_path: Path) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train Accuracy")
    axes[1].plot(epochs, history["val_acc"], label="Val Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_confusion_matrix(cm: np.ndarray, class_names: Sequence[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def classify_pattern(
    posx: Iterable[float],
    posy: Iterable[float],
    energy: Iterable[float],
    model_path: str = "laue_classifier.pth",
) -> tuple[str, float]:
    torch, _, _, _, _, models = _load_torch()
    checkpoint = torch.load(model_path, map_location="cpu")
    class_names = checkpoint["class_names"]

    try:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(weights=None)

    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    image = render_pattern(
        posx,
        posy,
        energy,
        detector_dim=checkpoint.get("detector_dim", DETECTOR_DIM),
        output_size=checkpoint.get("output_size", OUTPUT_SIZE),
        spot_radius=checkpoint.get("spot_radius", SPOT_RADIUS),
    )

    image = np.clip(image / checkpoint.get("max_energy_kev", ENERGY_BANDPASS_KEV[1]), 0.0, 1.0)
    image = np.repeat(image[None, :, :], 3, axis=0)
    tensor = torch.from_numpy(image).float().unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())
        confidence = float(probs[pred_idx].item())

    return class_names[pred_idx], confidence


def main() -> None:
    parser = argparse.ArgumentParser(description="LaueTools CNN dataset and classifier pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen_parser = subparsers.add_parser("generate", help="Generate Laue pattern dataset")
    gen_parser.add_argument("--output-dir", type=Path, default=Path("training_data"))
    gen_parser.add_argument("--samples-per-material", type=int, default=500)
    gen_parser.add_argument("--seed", type=int, default=0)

    train_parser = subparsers.add_parser("train", help="Train a ResNet18 classifier")
    train_parser.add_argument("--data-dir", type=Path, default=Path("training_data"))
    train_parser.add_argument("--model-path", type=Path, default=Path("laue_classifier.pth"))
    train_parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--learning-rate", type=float, default=1e-4)
    train_parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    if args.command == "generate":
        generate_dataset(args.output_dir, samples_per_material=args.samples_per_material, seed=args.seed)
    elif args.command == "train":
        train_classifier(
            args.data_dir,
            args.model_path,
            args.outputs_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
