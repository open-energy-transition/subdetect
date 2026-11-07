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
    if cfg.get("seed") is not None:
        # Paired A/B arms (docs/issues/esri-core-labels.md) share a seed so they start from
        # the same decoder init and see the same data order; the label is the only change.
        # Unset = historical behaviour (unseeded).
        from lightning import seed_everything

        seed_everything(int(cfg["seed"]), workers=True)

    dm = SubDataModule(**cfg["data"])
    task_args = dict(cfg["task"])
    task_type = cfg.get("task_type", "segmentation")
    if task_type == "regression":
        from terratorch.tasks import PixelwiseRegressionTask

        if task_args.get("loss") == "weighted_mse":
            # Counters the zero-inflation of a fraction target; see subdetect.losses. Like
            # the tversky branch below, PixelwiseRegressionTask accepts a loss nn.Module.
            from subdetect.losses import TargetWeightedMSE

            wa = task_args.pop("weighted_mse_args", None) or {}
            task_args["loss"] = TargetWeightedMSE(
                k=wa.get("k", 10.0),
                ignore_index=task_args.get("ignore_index", -1),
                under_weight=wa.get("under_weight", 1.0),
            )
        task = PixelwiseRegressionTask(**task_args)
    else:
        if task_args.get("loss") == "tversky":
            # TerraTorch has no built-in Tversky, but SemanticSegmentationTask accepts a
            # loss nn.Module. Tversky with beta>alpha penalises false negatives (missed
            # substations) harder than false positives -> recall-first. With a module loss
            # the task's class_weights is inactive (alpha/beta do the class weighting), so
            # drop it.
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
    try:
        import segmentation_models_pytorch as smp

        from subdetect.losses import TargetWeightedMSE

        torch.serialization.add_safe_globals([smp.losses.TverskyLoss, TargetWeightedMSE])
    except Exception:  # noqa: BLE001
        pass
    resume_ckpt = ckpt_dir / "last.ckpt"
    if resume_ckpt.exists():
        log.info("Resuming interrupted run from %s", resume_ckpt)
    elif cfg.get("init_weights_from"):
        # Warm-start from a previously fine-tuned checkpoint's weights only (fresh
        # optimizer/epoch count) -- e.g. continuing training with newly added data.
        state = torch.load(cfg["init_weights_from"], map_location="cpu")["state_dict"]
        task.load_state_dict(state)
        log.info("Initialized weights from %s", cfg["init_weights_from"])
    # Default to foreground IoU (val/IoU_1), not mIoU: mIoU averages in the background
    # class's IoU, which sits near 1.0 on a mostly-negative val split and swamps the
    # substation-detection signal we actually care about. Overridable via config.
    # PixelwiseRegressionTask logs no IoU at all (val/RMSE, val/MSE, val/MAE, val/R2_Score,
    # val/loss), so the regression default is RMSE-minimising.
    if task_type == "regression":
        monitor = cfg.get("monitor", "val/RMSE")
        mode = cfg.get("monitor_mode", "min")
    else:
        monitor = cfg.get("monitor", "val/IoU_1")
        mode = cfg.get("monitor_mode", "max")
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir, filename="terramind-sub-{epoch:02d}-{step}",
            monitor=monitor, save_top_k=2, mode=mode, save_last=True,
        ),
    ]
    # `early_stopping: false` trains to a fixed budget (max_epochs) under a schedule that
    # decays to its end, and the model is the final weights (last.ckpt), not an epoch picked
    # on the 289-chip val set, whose val/IoU_1 swings 0.06-0.26 between neighbouring epochs.
    # The recipe is earthpv's v22 (warmup + cosine + weight EMA), which ended the same epoch
    # lottery there. Every run up to v17 stops after `patience` epochs instead.
    if cfg.get("early_stopping", True):
        callbacks.append(EarlyStopping(monitor=monitor, patience=cfg.get("patience", 8),
                                       mode=mode))
    if cfg.get("ema"):
        # Exponential moving average of the weights: validation runs on the average and every
        # checkpoint's state_dict holds it (the raw weights ride along), so inference loads
        # the averaged model unchanged and a resumed run restores both.
        from lightning.pytorch.callbacks import EMAWeightAveraging

        callbacks.append(EMAWeightAveraging(decay=float(cfg["ema"].get("decay", 0.999))))
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
    if not cfg.get("early_stopping", True) and not smoke and not trainer.interrupted:
        # A fixed-budget run's model is its final state, so write it here: Lightning 2.6's
        # ModelCheckpoint refreshes last.ckpt only at a step where it also saves a top-k
        # checkpoint (earthpv v22 seed 1's last.ckpt was left one epoch short, 2026-10-31).
        trainer.save_checkpoint(ckpt_dir / "last.ckpt")
    best = callbacks[0].best_model_path or str(ckpt_dir / "last.ckpt")
    log.info("Best checkpoint: %s", best)
    return Path(best)
