"""Fine-tune TerraMind for substation segmentation via TerraTorch's SemanticSegmentationTask."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

log = logging.getLogger(__name__)


def run_training(config: Path, smoke: bool = False) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import torch
    from lightning import Trainer
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from terratorch.tasks import SemanticSegmentationTask

    from subdetect.datamodule import SubDataModule

    cfg = yaml.safe_load(Path(config).read_text())
    torch.set_float32_matmul_precision("medium")

    dm = SubDataModule(**cfg["data"])
    task_args = dict(cfg["task"])
    if task_args.get("loss") == "tversky":
        # TerraTorch has no built-in Tversky, but SemanticSegmentationTask accepts a loss
        # nn.Module. Tversky with beta>alpha penalises false negatives (missed substations)
        # harder than false positives -> recall-first. With a module loss the task's
        # class_weights is inactive (alpha/beta do the class weighting), so drop it.
        import segmentation_models_pytorch as smp

        ta = task_args.pop("tversky_args", None) or {}
        task_args.pop("class_weights", None)
        task_args["loss"] = smp.losses.TverskyLoss(
            mode="multiclass",
            ignore_index=task_args.get("ignore_index", -1),
            alpha=ta.get("alpha", 0.3),
            beta=ta.get("beta", 0.7),
            gamma=ta.get("gamma", 1.0),
        )
    task = SemanticSegmentationTask(**task_args)

    ckpt_dir = Path(cfg.get("checkpoint_dir", "data/models"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Auto-resume takes priority: this environment kills long background jobs
    # unpredictably, so a same-config restart must pick up mid-training rather than
    # re-warm-start. Lightning's ckpt_path restores model+optimizer+epoch+callbacks.
    resume_ckpt = ckpt_dir / "last.ckpt"
    if resume_ckpt.exists():
        log.info("Resuming interrupted run from %s", resume_ckpt)
    elif cfg.get("init_weights_from"):
        # Warm-start from a previously fine-tuned checkpoint's weights only (fresh
        # optimizer/epoch count) -- e.g. continuing training with newly added data.
        try:
            import segmentation_models_pytorch as smp

            torch.serialization.add_safe_globals([smp.losses.TverskyLoss])
        except Exception:  # noqa: BLE001
            pass
        state = torch.load(cfg["init_weights_from"], map_location="cpu")["state_dict"]
        task.load_state_dict(state)
        log.info("Initialized weights from %s", cfg["init_weights_from"])
    # Recall-first: monitor balanced (macro) recall = val/Accuracy (MulticlassAccuracy
    # macro = mean per-class recall) instead of mIoU. Overridable via config.
    monitor = cfg.get("monitor", "val/mIoU")
    mode = cfg.get("monitor_mode", "max")
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir, filename="terramind-sub-{epoch:02d}-{step}",
            monitor=monitor, save_top_k=2, mode=mode, save_last=True,
        ),
        EarlyStopping(monitor=monitor, patience=cfg.get("patience", 8), mode=mode),
    ]
    trainer_kwargs = dict(cfg.get("trainer", {}))
    if smoke:
        trainer_kwargs.update(max_steps=50, val_check_interval=25, limit_val_batches=4,
                              max_epochs=None)
    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision="16-mixed",
        callbacks=callbacks,
        log_every_n_steps=5,
        default_root_dir="logs",
        **trainer_kwargs,
    )
    trainer.fit(task, datamodule=dm, ckpt_path=str(resume_ckpt) if resume_ckpt.exists() else None)
    best = callbacks[0].best_model_path or str(ckpt_dir / "last.ckpt")
    log.info("Best checkpoint: %s", best)
    return Path(best)
