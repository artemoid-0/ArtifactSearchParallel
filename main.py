import os
from collections import Counter
from pathlib import Path
import random

import numpy as np
import torch
from timm import optim
from torch import nn
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchmetrics.functional.classification import f1_score
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights, ResNet18_Weights, resnet18
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from core.dataset import CustomImageDataset


def ddp_setup():
    init_process_group(
        backend='gloo',
    )


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Trainer:
    def __init__(self, model, train_data, val_data, optimizer, scheduler, save_every, snapshot_path, best_model_path,
                 cm_path, pos_weight):
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.model = model.to(self.local_rank)
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        self.best_model_path = best_model_path
        self.cm_path = cm_path

        if snapshot_path.exists():
            print(f"Loading snapshot from {snapshot_path}")
            self._load_snapshot(snapshot_path)

        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to("cuda"))
        self.model = DDP(self.model, device_ids=[self.local_rank], find_unused_parameters=True)

    def _load_snapshot(self, snapshot_path):
        snapshot = torch.load(snapshot_path)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output, aux_output = self.model(source)  # output: [B, 1], aux_output: [B, 1]

        # Combine main and auxiliary loss if desired
        loss1 = self.loss_fn(output, targets.unsqueeze(1).float())
        loss2 = self.loss_fn(aux_output, targets.unsqueeze(1).float())
        loss = loss1 + 0.4 * loss2  # weighted sum as suggested in Inception paper

        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _run_epoch(self, epoch):
        if isinstance(self.train_data.sampler, torch.utils.data.DistributedSampler):
            self.train_data.sampler.set_epoch(epoch)

        self.model.train()
        total_loss = 0
        for source, target, _ in self.train_data:
            source = source.to(device=self.local_rank)
            targets = target.to(device=self.local_rank)
            loss = self._run_batch(source, targets)
            total_loss += loss
        avg_loss = total_loss / len(self.train_data)
        b_sz = len(next(iter(self.train_data))[0])
        print(
            f"[GPU{self.global_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)} | Train Loss: {avg_loss:.4f}")
        return avg_loss

    def _validate(self):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for source, target, _ in self.val_data:
                source = source.to(device=self.local_rank)
                target = target.to(device=self.local_rank)

                output = self.model(source)  # [B, 1]
                if isinstance(output, tuple):  # if aux_logits still active
                    output = output[0]

                loss = self.loss_fn(output, target.unsqueeze(1).float())
                total_loss += loss.item()

                probs = torch.sigmoid(output).squeeze(1)  # [B]
                preds = (probs > 0.5).long()  # binary prediction

                all_preds.append(preds.cpu())
                all_targets.append(target.cpu())

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        f1 = f1_score(all_preds, all_targets, task="binary").item()
        avg_loss = total_loss / len(self.val_data)
        print(f"[GPU{self.local_rank}] Validation Loss: {avg_loss:.4f} | F1 Score: {f1:.4f}")
        return avg_loss, f1

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        best_f1 = 0.0
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            val_loss, val_f1 = self._validate()

            if self.local_rank == 0:
                if self.scheduler:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f"[GPU{self.local_rank}] Current LR: {current_lr}")

                if val_f1 > best_f1:
                    best_f1 = val_f1
                    self._save_snapshot(epoch)
                    torch.save(self.model.module.state_dict(), self.best_model_path)
                    print(f"[GPU{self.local_rank}] Best model saved to {self.best_model_path} (F1: {best_f1:.4f})")

    def test_model(self, test_dataloader):
        print(f"[GPU{self.local_rank}] Testing model on test set...")
        _, _, preds, targets, filenames, probs = self._evaluate(test_dataloader, name="Test", return_outputs=True,
                                                                log_filenames=True)
        if self.local_rank == 0:
            self._plot_confusion_matrix(preds, targets)
            self._save_test_report(preds, targets, filenames, probs)

    def _save_test_report(self, preds, targets, filenames, probs):
        report_path = self.cm_path.parent / "test_report.csv"
        with open(report_path, "w") as f:
            f.write("filename,true_label,predicted_label,correct,prob_artifact\n")
            for name, true, pred, prob in zip(filenames, targets, preds, probs):
                correct = "yes" if true == pred else "no"
                f.write(f"{name},{true.item()},{pred.item()},{correct},{prob.item():.4f}\n")
        print(f"[GPU0] Per-sample test report saved at {report_path}")

    def _plot_confusion_matrix(self, preds, targets):
        cm = confusion_matrix(targets, preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Class 0", "Class 1"],
                    yticklabels=["Class 0", "Class 1"])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix on Test Set")
        plt.tight_layout()
        plt.savefig(self.cm_path)
        print(f"[GPU0] Confusion matrix saved at {self.cm_path}")

    def _evaluate(self, dataloader, name="Validation", return_outputs=False, log_filenames=False):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_filenames = []
        all_probs = []

        with torch.no_grad():
            for batch in dataloader:
                if log_filenames:
                    source, target, filenames = batch
                    all_filenames.extend(filenames)
                else:
                    source, target = batch

                source = source.to(device=self.local_rank)
                target = target.to(device=self.local_rank)

                output = self.model(source)
                if isinstance(output, tuple):  # if aux_logits still active
                    output = output[0]

                loss = self.loss_fn(output, target.unsqueeze(1).float())
                total_loss += loss.item()

                probs = torch.sigmoid(output).squeeze(1)  # [B]
                preds = (probs > 0.5).long()  # binary prediction

                all_preds.append(preds.cpu())
                all_targets.append(target.cpu())
                all_probs.append(probs.cpu())

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        all_probs = torch.cat(all_probs)

        f1 = f1_score(all_preds, all_targets, task="binary").item()
        avg_loss = total_loss / len(dataloader)
        print(f"[GPU{self.local_rank}] {name} Loss: {avg_loss:.4f} | F1 Score: {f1:.4f}")

        if return_outputs:
            if log_filenames:
                return avg_loss, f1, all_preds, all_targets, all_filenames, all_probs
            return avg_loss, f1, all_preds, all_targets

        return avg_loss, f1


def compute_pos_weight_from_dataset(dataset):
    labels = [label for _, label, _ in dataset]
    label_tensor = torch.tensor(labels)
    num_pos = (label_tensor == 1).sum().item()
    num_neg = (label_tensor == 0).sum().item()
    if num_pos == 0 or num_neg == 0:
        raise ValueError("Dataset must contain both positive and negative samples.")
    pos_weight = num_neg / num_pos
    return torch.tensor([pos_weight], dtype=torch.float32)


def load_train_objs():
    image_size = 299

    # Paths
    # train_path = r"D:\DATASETS\artifact_dataset\train"
    # test_path = r"D:\DATASETS\artifact_dataset\test"
    train_path = Path(__file__).resolve().parent / "face_focus" / "GOOD_RESERVE" / "faces_cropped_mtcnn_(train)"
    test_path = Path(__file__).resolve().parent / "face_focus" / "GOOD_RESERVE" / "faces_cropped_mtcnn_(test)"

    # Transformations
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the datasets
    full_train_dataset = CustomImageDataset(str(train_path), transform=train_transform, flip_labels=False,
                                            labels_csv=r"D:\DATASETS\artifact_dataset\labels\gaze_ignoring\train_labels.csv"
                                            )
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    test_dataset = CustomImageDataset(str(test_path), transform=test_transform, flip_labels=False,
                                      labels_csv=r"D:\DATASETS\artifact_dataset\labels\gaze_ignoring\test_labels.csv"
                                      )

    # Inside your load_train_objs() function, after full_train_dataset is loaded:
    pos_weight = compute_pos_weight_from_dataset(full_train_dataset)
    print(f"Computed pos_weight: {pos_weight.item():.4f}")

    # Class distribution functions
    def print_class_distribution(dataset, dataset_name):
        # Get class labels for the dataset
        class_labels = [label for _, label, _ in dataset]
        class_counts = Counter(class_labels)

        print(f"\nClass distribution for {dataset_name}:")
        for class_idx, count in class_counts.items():
            print(f"Class {class_idx}: {count} samples")

    # Print class distribution for train and val datasets
    # print_class_distribution(train_dataset, "train dataset")
    # print_class_distribution(val_dataset, "validation dataset")

    # Model and optimizer
    model = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True)
    model.fc = nn.Linear(model.fc.in_features, 1)  # binary classification
    model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, 1)  # optional if aux is used

    # model = resnet18(weights=ResNet18_Weights.DEFAULT)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    return train_dataset, val_dataset, test_dataset, model, optimizer, scheduler, pos_weight


def prepare_dataloader(dataset: torch.utils.data.Dataset, batch_size: int,
                       shuffle: bool = False) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        sampler=DistributedSampler(dataset, shuffle=shuffle)
    )


def main(run_name: str, save_every: int, total_epochs: int):
    set_seed(42)
    ddp_setup()

    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = run_dir / "snapshot.pt"
    best_model_path = run_dir / "best_model.pt"
    confusion_matrix_path = run_dir / "confusion_matrix.png"

    train_dataset, val_dataset, test_dataset, model, optimizer, scheduler, pos_weight = load_train_objs()
    train_data = prepare_dataloader(train_dataset, batch_size=16, shuffle=True)
    val_data = prepare_dataloader(val_dataset, batch_size=16, shuffle=False)
    test_data = prepare_dataloader(test_dataset, batch_size=16, shuffle=False)

    trainer = Trainer(model, train_data, val_data, optimizer, scheduler, save_every, snapshot_path, best_model_path,
                      confusion_matrix_path, pos_weight)
    trainer.train(total_epochs)

    # Test best model
    if trainer.local_rank == 0 and best_model_path.exists():
        trainer.model.module.load_state_dict(torch.load(best_model_path))
        print("[GPU0] Loaded best model for final test evaluation")
    trainer.test_model(test_data)

    destroy_process_group()


if __name__ == '__main__':
    run_name = "experiment_012 (gaze_ignoring_dataset, inception_v3, wieght_BCE_loss, batch size=16, faces_only)"
    total_epochs = 20
    save_every = 2

    main(run_name, save_every, total_epochs)
