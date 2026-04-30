"""
train_model.py (v2)
==============
TerraTorch / Prithvi / timm vision model for regressing TrueSkill wildland scores.
Supports U-Net, Xception (no pooling), and TerraMind foundation models.

Supported Backbones:
--------------------
1. Generic timm: --backbone resnet18, efficientnet_b0, etc.
2. Xception (no pooling): --backbone xception
3. U-Net (Encoder part for regression): --backbone unet
4. TerraMind (TerraTorch): --use_prithvi --prithvi_model terramind_v1_base

Loss: configurable (--loss); default GNLL (predicts mean and variance).
LR schedulers: --lr_scheduler none|cosine|reduce_on_plateau|onecycle.
HDF5: standard NCHW + attrs, or NHWC tile exports (wildness + mean lat/lon target).
Uncertainty: MC Dropout.
"""

import argparse, os, math, re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import h5py
import timm
import albumentations as A
from tqdm import tqdm


# ── CLI ───────────────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser(description="Train GeoVisionModel")
    p.add_argument("--hdf5",        default="dataset.h5")
    p.add_argument("--val_hdf5",    default="",
                   help="Optional separate validation HDF5. If set, --val_split is ignored.")
    p.add_argument("--val_split",   type=float, default=0.15)
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--batch_size",  type=int,   default=16)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--seed",        type=int,   default=42)

    p.add_argument("--backbone",    default="resnet18",
                   help="timm model name. Special cases: 'xception', 'unet'")
    p.add_argument("--dropout",     type=float, default=0.3)

    p.add_argument("--use_prithvi", action="store_true")
    p.add_argument("--prithvi_model",
                   default="ibm-nasa-geospatial/Prithvi-EO-1.0-100M")
    p.add_argument("--prithvi_bands", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument(
        "--finetune_mode",
        default="full",
        choices=("full", "freeze_backbone", "freeze_partial"),
        help="TerraMind finetune strategy. Ignored when --use_prithvi is not set.",
    )
    p.add_argument(
        "--freeze_partial_depth",
        type=int,
        default=8,
        help="For freeze_partial: freeze encoder layers with index < depth.",
    )
    p.add_argument(
        "--unfreeze_epoch",
        type=int,
        default=0,
        help="If > 0, unfreeze TerraMind backbone at this epoch (staged unfreeze).",
    )
    p.add_argument(
        "--llrd_gamma",
        type=float,
        default=1.0,
        help="Layer-wise LR decay gamma for TerraMind backbone (<1 enables decay).",
    )

    p.add_argument("--mc_samples",  type=int,   default=20)
    p.add_argument("--amp", action="store_true",
                   help="Enable mixed precision training/inference (CUDA autocast + GradScaler).")
    p.add_argument("--log_dir",     default="runs/wildterrain")
    p.add_argument("--ckpt_dir",    default="checkpoints")
    p.add_argument("--resume_ckpt", default="",
                   help="Optional checkpoint path to resume model weights from.")
    p.add_argument("--start_epoch", type=int, default=1,
                   help="Epoch index to start/resume from (1-based).")
    p.add_argument("--use_observed_std_in_loss", action="store_true",
                   help="Alias: same as --loss gnll_observed_std when --loss is gnll.")
    p.add_argument("--use_std_as_input", action="store_true",
                   help="Append score_std as an additional model input feature.")
    p.add_argument("--dry_run",     action="store_true")

    p.add_argument(
        "--loss",
        default="gnll",
        choices=("gnll", "gnll_observed_std", "mse", "mae", "smooth_l1"),
        help="Training objective. gnll_observed_std uses HDF5 score_std as label noise.",
    )
    p.add_argument(
        "--lr_scheduler",
        default="none",
        choices=("none", "cosine", "reduce_on_plateau", "onecycle"),
    )
    p.add_argument("--lr_min", type=float, default=1e-6,
                   help="Floor LR for cosine / OneCycle min_lr.")
    p.add_argument("--scheduler_patience", type=int, default=5,
                   help="ReduceLROnPlateau patience (epochs).")
    p.add_argument("--scheduler_factor", type=float, default=0.5,
                   help="ReduceLROnPlateau factor.")
    p.add_argument("--onecycle_max_lr", type=float, default=None,
                   help="OneCycle max LR; default = 10 * --lr.")
    p.add_argument(
        "--metric_for_best",
        default="loss",
        choices=("loss", "gnll", "mse", "mae", "rmse"),
        help="Validation metric to minimize for best_model.pt (gnll = standard GNLL diagnostic).",
    )
    return p.parse_args()


def apply_loss_flag_aliases(args):
    if args.use_observed_std_in_loss and args.loss == "gnll":
        args.loss = "gnll_observed_std"


# ── HDF5 layout ───────────────────────────────────────────────────────────────
def infer_hdf5_layout(hdf5_path):
    """
    Returns dict: layout 'standard'|'nhwc_agg', n_bands, img_size, has_score_std,
    target_key ('score_norm'|'wildness').
    """
    with h5py.File(hdf5_path, "r") as f:
        im = f["images"]
        sh = im.shape
        if "n_bands" in f.attrs and "img_size" in f.attrs:
            return {
                "layout": "standard",
                "n_bands": int(f.attrs["n_bands"]),
                "img_size": int(f.attrs["img_size"]),
                "has_score_std": "score_std" in f,
                "target_key": "score_norm",
            }
        # Heuristic: (N, H, W, C) tile export with wildness / per-pixel geo
        if len(sh) == 4 and sh[-1] <= 32 and sh[1] == sh[2] and sh[-1] >= 3:
            if "wildness" not in f:
                raise ValueError(
                    f"HDF5 {hdf5_path} missing root attrs n_bands/img_size and no 'wildness' for NHWC layout."
                )
            return {
                "layout": "nhwc_agg",
                "n_bands": int(sh[-1]),
                "img_size": int(sh[1]),
                "has_score_std": "score_std" in f,
                "target_key": "wildness",
            }
        if len(sh) == 4 and sh[1] <= 32:  # NCHW without attrs
            return {
                "layout": "standard",
                "n_bands": int(sh[1]),
                "img_size": int(sh[2]),
                "has_score_std": "score_std" in f,
                "target_key": "score_norm",
            }
    raise ValueError(f"Cannot infer layout for images shape {sh} in {hdf5_path}")


# ── Dataset ───────────────────────────────────────────────────────────────────
class HDF5Dataset(Dataset):
    TRAIN_AUG = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(p=0.2),
    ])

    def __init__(self, hdf5_path, indices=None, augment=False, layout_info=None):
        self.path = hdf5_path
        self.augment = augment
        self.layout = layout_info["layout"]
        self.target_key = layout_info["target_key"]
        with h5py.File(hdf5_path, "r") as f:
            self.n_total = len(f["images"])
            self.has_score_std = layout_info["has_score_std"]
        self.indices = indices if indices is not None else list(range(self.n_total))
        self._file = None

    def _open(self):
        if self._file is None:
            self._file = h5py.File(self.path, "r", swmr=True)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        self._open()
        i = self.indices[idx]

        if self.layout == "standard":
            img = self._file["images"][i]
            score = float(self._file["score_norm"][i])
            score_std = float(self._file["score_std"][i]) if self.has_score_std else 0.0
            lat = float(self._file["lat"][i])
            lon = float(self._file["lon"][i])
        else:
            # NHWC uint16/float images; aggregate wildness + mean lat/lon
            img_hwc = self._file["images"][i]
            if img_hwc.dtype == np.uint16:
                img_hwc = img_hwc.astype(np.float32) / 10000.0
            else:
                img_hwc = img_hwc.astype(np.float32)
            w = self._file["wildness"][i]
            score = float(np.mean(w))
            score_std = float(np.std(w)) if w.size > 1 else 0.0
            lat = float(np.mean(self._file["lat"][i]))
            lon = float(np.mean(self._file["lon"][i]))
            img = np.transpose(img_hwc, (2, 0, 1))

        if self.augment:
            img_hwc = np.transpose(img, (1, 2, 0))
            img_hwc = self.TRAIN_AUG(image=img_hwc)["image"]
            img = np.transpose(img_hwc, (2, 0, 1))

        img_t = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32))
        geo_t = torch.tensor([
            math.sin(math.radians(lat)), math.cos(math.radians(lat)),
            math.sin(math.radians(lon)), math.cos(math.radians(lon)),
        ], dtype=torch.float32)
        return (
            img_t,
            geo_t,
            torch.tensor(score, dtype=torch.float32),
            torch.tensor(score_std, dtype=torch.float32),
        )


# ── MC Dropout ───────────────────────────────────────────────────────────────
class MCDropout(nn.Module):
    def __init__(self, p=0.3):
        super().__init__(); self.p = p
    def forward(self, x):
        return nn.functional.dropout(x, p=self.p, training=True)


# ── Backbones ─────────────────────────────────────────────────────────────────
class PrithviBackbone(nn.Module):
    def __init__(self, model_id, selected_bands, img_size=224):
        super().__init__()
        self.selected_bands = selected_bands
        try:
            from terratorch.models.backbones.prithvi_mae import PrithviViT
            self.encoder = PrithviViT(img_size=img_size, num_frames=1, in_chans=6)
            self.feat_dim = self.encoder.embed_dim
            self._use_terratorch = True
        except ImportError:
            print("[Warning] terratorch not found, falling back to timm/HF for foundation model...")
            self._use_terratorch = False
            self.encoder = timm.create_model("vit_base_patch16_224", pretrained=False, in_chans=len(selected_bands), num_classes=0)
            self.feat_dim = self.encoder.num_features

    def forward(self, x):
        if self._use_terratorch:
            bands = x[:, self.selected_bands, :, :]
            c = bands.shape[1]
            repeats = math.ceil(6 / c) if c > 0 else 6
            bands6 = bands.repeat(1, repeats, 1, 1)[:, :6, :, :].unsqueeze(2)
            enc = self.encoder
            if hasattr(enc, "forward_features"):
                latent = enc.forward_features(bands6)[-1]
            else:
                latent, _, _ = enc.forward_encoder(bands6, mask_ratio=0.0)
            return latent[:, 1:, :].mean(dim=1)
        return self.encoder(x[:, self.selected_bands, :, :])


def get_timm_backbone(name, n_bands, img_size=224):
    if name == "xception":
        model = timm.create_model("xception", pretrained=False, in_chans=n_bands, num_classes=0, global_pool='')
        # Xception runs without global pooling, so flattened feature size depends on image resolution.
        dummy = torch.randn(1, n_bands, img_size, img_size)
        feat_dim = model(dummy).view(1, -1).shape[1]
        return model, feat_dim, True
    elif name == "unet":
        model = timm.create_model("resnet34", pretrained=False, in_chans=n_bands, num_classes=0)
        return model, model.num_features, False
    else:
        model = timm.create_model(name, pretrained=False, in_chans=n_bands, num_classes=0, global_pool='avg')
        return model, model.num_features, False


# ── Model ─────────────────────────────────────────────────────────────────────
class GeoVisionModel(nn.Module):
    def __init__(self, args, n_bands, img_size):
        super().__init__()
        self.dropout_p = args.dropout

        if args.use_prithvi:
            self.backbone = PrithviBackbone(args.prithvi_model, args.prithvi_bands, img_size)
            self.feat_dim = self.backbone.feat_dim
            self.needs_flatten = False
        else:
            self.backbone, self.feat_dim, self.needs_flatten = get_timm_backbone(
                args.backbone, n_bands, img_size=img_size
            )

        self.mc_drop = MCDropout(p=args.dropout)

        fusion_dim = self.feat_dim + 4 + (1 if args.use_std_as_input else 0)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            MCDropout(p=args.dropout),
            nn.Linear(256, 128),
            nn.GELU(),
        )
        self.head_mean = nn.Linear(128, 1)
        self.head_log_var = nn.Linear(128, 1)

    def forward(self, img, geo, score_std=None):
        feats = self.backbone(img)
        if self.needs_flatten:
            feats = feats.view(feats.size(0), -1)
        feats = self.mc_drop(feats)
        if score_std is not None:
            fused = torch.cat([feats, geo, score_std.unsqueeze(1)], dim=1)
        else:
            fused = torch.cat([feats, geo], dim=1)
        h = self.fusion_mlp(fused)
        return self.head_mean(h).squeeze(1), self.head_log_var(h).squeeze(1)


# ── Loss / metrics ───────────────────────────────────────────────────────────
def gnll_loss(mean, log_var, target):
    var = torch.exp(log_var) + 1e-6
    return (0.5 * (log_var + (target - mean)**2 / var)).mean()


def gnll_loss_with_observed_std(mean, log_var, target, observed_std):
    model_var = torch.exp(log_var).clamp_min(1e-6)
    obs_var = observed_std.pow(2).clamp_min(1e-6)
    total_var = model_var + obs_var
    return (0.5 * (torch.log(total_var) + (target - mean) ** 2 / total_var)).mean()


def train_step_loss(args, m, lv, targets, target_stds):
    if args.loss == "gnll":
        return gnll_loss(m, lv, targets)
    if args.loss == "gnll_observed_std":
        return gnll_loss_with_observed_std(m, lv, targets, target_stds)
    err = m - targets
    if args.loss == "mse":
        return err.pow(2).mean()
    if args.loss == "mae":
        return err.abs().mean()
    if args.loss == "smooth_l1":
        return nn.functional.smooth_l1_loss(m, targets)
    raise RuntimeError(f"Unknown loss {args.loss}")


def state_dict_for_save(model):
    return model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()


def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def terramind_layer_id_from_name(param_name):
    """
    Best-effort depth extraction for ViT-style and staged backbones.
    Returns 0 for embeddings/non-block params.
    """
    m = re.search(r"(?:blocks|layers)\.(\d+)", param_name)
    if m:
        return int(m.group(1)) + 1
    return 0


def apply_terramind_finetune_mode(model, args, mode):
    """
    Set requires_grad for TerraMind backbone according to mode.
    """
    base = unwrap_model(model)
    if not args.use_prithvi or not hasattr(base.backbone, "encoder"):
        return

    encoder = base.backbone.encoder
    if mode == "full":
        for p in encoder.parameters():
            p.requires_grad = True
        return

    if mode == "freeze_backbone":
        for p in encoder.parameters():
            p.requires_grad = False
        return

    if mode == "freeze_partial":
        depth = max(0, int(args.freeze_partial_depth))
        for n, p in encoder.named_parameters():
            layer_id = terramind_layer_id_from_name(n)
            p.requires_grad = layer_id >= depth
        return

    raise RuntimeError(f"Unknown finetune_mode: {mode}")


def build_optimizer_param_groups(model, args):
    """
    Build optimizer groups with optional TerraMind layer-wise LR decay.
    """
    if not args.use_prithvi:
        return [{"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr}]

    base = unwrap_model(model)
    if not hasattr(base.backbone, "encoder"):
        return [{"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr}]

    enc_named = list(base.backbone.encoder.named_parameters())
    head_named = []
    head_named.extend(base.mc_drop.named_parameters())
    head_named.extend(base.fusion_mlp.named_parameters())
    head_named.extend(base.head_mean.named_parameters())
    head_named.extend(base.head_log_var.named_parameters())

    groups = []
    llrd_on = args.llrd_gamma < 1.0
    gamma = max(1e-6, float(args.llrd_gamma))

    if llrd_on:
        layer_to_params = {}
        max_layer = 0
        for n, p in enc_named:
            layer_id = terramind_layer_id_from_name(n)
            max_layer = max(max_layer, layer_id)
            layer_to_params.setdefault(layer_id, []).append(p)
        for layer_id, params in sorted(layer_to_params.items(), key=lambda kv: kv[0]):
            lr_scale = gamma ** (max_layer - layer_id)
            groups.append({"params": params, "lr": args.lr * lr_scale})
    else:
        groups.append({"params": [p for _, p in enc_named], "lr": args.lr})

    groups.append({"params": [p for _, p in head_named], "lr": args.lr})
    return groups


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_scheduler(optimizer, args, steps_per_epoch):
    sch = args.lr_scheduler
    if sch == "none":
        return None
    if sch == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs), eta_min=args.lr_min
        )
    if sch == "reduce_on_plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
            min_lr=args.lr_min,
        )
    if sch == "onecycle":
        max_lr = args.onecycle_max_lr if args.onecycle_max_lr is not None else args.lr * 10.0
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=args.epochs,
            steps_per_epoch=max(1, steps_per_epoch),
            final_div_factor=max(1.0, args.lr / max(args.lr_min, 1e-12)),
        )
    raise RuntimeError(f"Unknown scheduler {sch}")


def metric_bundle(avg_loss, avg_gnll, avg_mae, avg_mse, avg_rmse):
    return {
        "loss": avg_loss,
        "gnll": avg_gnll,
        "mse": avg_mse,
        "mae": avg_mae,
        "rmse": avg_rmse,
    }


def main():
    args = get_args()
    apply_loss_flag_aliases(args)
    torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        n_gpu = torch.cuda.device_count()
        for i in range(n_gpu):
            print(f"[cuda:{i}] {torch.cuda.get_device_name(i)}")
        print(f"Using device {device} ({n_gpu} GPU(s) visible; DataParallel if n_gpu>1)")
    else:
        print("CUDA not available — training on CPU.")
    amp_enabled = bool(args.amp and use_cuda)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    train_hdf5 = args.hdf5
    val_hdf5 = args.val_hdf5.strip()

    train_layout = infer_hdf5_layout(train_hdf5)
    n_bands, img_size = train_layout["n_bands"], train_layout["img_size"]
    with h5py.File(train_hdf5, "r") as f:
        n_total = len(f["images"])

    if val_hdf5:
        val_layout = infer_hdf5_layout(val_hdf5)
        if val_layout["n_bands"] != n_bands:
            raise ValueError(
                f"Train/val HDF5 n_bands mismatch: train={n_bands}, val={val_layout['n_bands']}"
            )
        if val_layout["img_size"] != img_size:
            raise ValueError(
                f"Train/val HDF5 img_size mismatch: train={img_size}, val={val_layout['img_size']}"
            )
        with h5py.File(val_hdf5, "r") as f:
            n_val_total = len(f["images"])
        train_ds = HDF5Dataset(train_hdf5, augment=True, layout_info=train_layout)
        val_ds = HDF5Dataset(val_hdf5, augment=False, layout_info=val_layout)
        print(
            f"Train HDF5={train_hdf5} n={n_total} layout={train_layout['layout']} target={train_layout['target_key']} | "
            f"Val HDF5={val_hdf5} n={n_val_total} layout={val_layout['layout']} target={val_layout['target_key']} | "
            f"n_bands={n_bands} img_size={img_size} loss={args.loss} scheduler={args.lr_scheduler} "
            f"metric_for_best={args.metric_for_best}"
        )
    else:
        indices = np.random.default_rng(args.seed).permutation(n_total)
        n_val = int(n_total * args.val_split)
        train_ds = HDF5Dataset(train_hdf5, indices[n_val:], augment=True, layout_info=train_layout)
        val_ds = HDF5Dataset(train_hdf5, indices[:n_val], augment=False, layout_info=train_layout)
        print(
            f"HDF5={train_hdf5} layout={train_layout['layout']} n_bands={n_bands} img_size={img_size} "
            f"target={train_layout['target_key']} split=single_file val_split={args.val_split} "
            f"loss={args.loss} scheduler={args.lr_scheduler} metric_for_best={args.metric_for_best}"
        )

    train_loader = DataLoader(
        train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=use_cuda,
    )
    val_loader = DataLoader(
        val_ds, args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=use_cuda,
    )

    model = GeoVisionModel(args, n_bands, img_size).to(device)
    if use_cuda and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    if args.resume_ckpt:
        ckpt = torch.load(args.resume_ckpt, map_location=device)
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(ckpt)
        else:
            model.load_state_dict(ckpt)
        print(f"Resumed model weights from: {args.resume_ckpt}")

    initial_mode = args.finetune_mode if args.use_prithvi else "full"
    apply_terramind_finetune_mode(model, args, initial_mode)
    total_params, trainable_params = count_params(model)
    llrd_active = args.use_prithvi and args.llrd_gamma < 1.0
    print(
        f"Finetune mode={initial_mode} | staged_unfreeze_epoch={args.unfreeze_epoch or 0} | "
        f"llrd_gamma={args.llrd_gamma:.4f} active={llrd_active} | "
        f"trainable_params={trainable_params:,}/{total_params:,} | amp={amp_enabled}"
    )

    optimizer = optim.AdamW(build_optimizer_param_groups(model, args), lr=args.lr)
    scheduler = build_scheduler(optimizer, args, len(train_loader))
    writer = SummaryWriter(log_dir=args.log_dir)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_val_metric = float("inf")
    start_epoch = max(1, args.start_epoch)
    unfreeze_applied = False
    for epoch in range(start_epoch, args.epochs + 1):
        if (
            args.use_prithvi
            and args.unfreeze_epoch > 0
            and epoch >= args.unfreeze_epoch
            and not unfreeze_applied
        ):
            apply_terramind_finetune_mode(model, args, "full")
            total_params, trainable_params = count_params(model)
            print(
                f"[StagedUnfreeze] epoch={epoch}: switched to full finetune, "
                f"trainable_params={trainable_params:,}/{total_params:,}"
            )
            unfreeze_applied = True

        model.train()
        t_loss = t_gnll = t_mae = t_mse = 0.0
        for imgs, geos, targets, target_stds in tqdm(train_loader, desc=f"Epoch {epoch}"):
            imgs = imgs.to(device)
            geos = geos.to(device)
            targets = targets.to(device)
            target_stds = target_stds.to(device)
            optimizer.zero_grad()
            std_input = target_stds if args.use_std_as_input else None
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                m, lv = model(imgs, geos, std_input)
                loss = train_step_loss(args, m, lv, targets, target_stds)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None and args.lr_scheduler == "onecycle":
                scheduler.step()

            t_loss += loss.item()
            t_gnll += gnll_loss(m, lv, targets).detach().item()
            err = m - targets
            t_mae += err.abs().mean().item()
            t_mse += err.pow(2).mean().item()

        n_t = len(train_loader)
        avg_t_loss = t_loss / n_t
        avg_t_gnll = t_gnll / n_t
        avg_t_mae = t_mae / n_t
        avg_t_mse = t_mse / n_t
        avg_t_rmse = math.sqrt(max(avg_t_mse, 0.0))

        model.eval()
        v_loss = v_gnll = v_mae = v_mse = 0.0
        v_aleatoric = v_epistemic = v_total = 0.0
        with torch.no_grad():
            for imgs, geos, targets, target_stds in val_loader:
                imgs = imgs.to(device)
                geos = geos.to(device)
                targets = targets.to(device)
                target_stds = target_stds.to(device)
                std_input = target_stds if args.use_std_as_input else None
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    outputs = [model(imgs, geos, std_input) for _ in range(args.mc_samples)]
                    preds_m = torch.stack([o[0] for o in outputs])
                    preds_lv = torch.stack([o[1] for o in outputs])
                    
                    m = preds_m.mean(0)
                    if args.mc_samples > 1:
                        epistemic_var = preds_m.var(0)
                    else:
                        epistemic_var = torch.zeros_like(m)
                        
                    aleatoric_var = torch.exp(preds_lv).mean(0)
                    total_var = aleatoric_var + epistemic_var
                    lv = torch.log(total_var)
                    
                    lv_det = train_step_loss(args, m, lv, targets, target_stds).item()
                v_loss += lv_det
                v_gnll += gnll_loss(m, lv, targets).item()
                err = m - targets
                v_mae += err.abs().mean().item()
                v_mse += err.pow(2).mean().item()
                v_aleatoric += aleatoric_var.mean().item()
                v_epistemic += epistemic_var.mean().item()
                v_total += total_var.mean().item()

        n_v = len(val_loader)
        avg_v_loss = v_loss / n_v
        avg_v_gnll = v_gnll / n_v
        avg_v_mae = v_mae / n_v
        avg_v_mse = v_mse / n_v
        avg_v_rmse = math.sqrt(max(avg_v_mse, 0.0))
        avg_v_aleatoric = v_aleatoric / n_v
        avg_v_epistemic = v_epistemic / n_v
        avg_v_total = v_total / n_v

        train_m = metric_bundle(avg_t_loss, avg_t_gnll, avg_t_mae, avg_t_mse, avg_t_rmse)
        val_m = metric_bundle(avg_v_loss, avg_v_gnll, avg_v_mae, avg_v_mse, avg_v_rmse)

        print(
            f"Epoch {epoch} | "
            f"Train loss({args.loss}): {avg_t_loss:.4f} | Val loss: {avg_v_loss:.4f} | "
            f"Train MSE: {avg_t_mse:.4f} MAE: {avg_t_mae:.4f} RMSE: {avg_t_rmse:.4f} | "
            f"Val MSE: {avg_v_mse:.4f} MAE: {avg_v_mae:.4f} RMSE: {avg_v_rmse:.4f} | "
            f"Train GNLL: {avg_t_gnll:.4f} Val GNLL: {avg_v_gnll:.4f} | "
            f"Val Var (Aleatoric: {avg_v_aleatoric:.4f}, Epistemic: {avg_v_epistemic:.4f}, Total: {avg_v_total:.4f})"
        )

        writer.add_scalar("Loss/train", avg_t_loss, epoch)
        writer.add_scalar("Loss/val", avg_v_loss, epoch)
        writer.add_scalar("OptimizedLoss/train", avg_t_loss, epoch)
        writer.add_scalar("OptimizedLoss/val", avg_v_loss, epoch)
        writer.add_scalar("GNLL/train", avg_t_gnll, epoch)
        writer.add_scalar("GNLL/val", avg_v_gnll, epoch)
        writer.add_scalar("Variance/val_aleatoric", avg_v_aleatoric, epoch)
        writer.add_scalar("Variance/val_epistemic", avg_v_epistemic, epoch)
        writer.add_scalar("Variance/val_total", avg_v_total, epoch)
        writer.add_scalar("MAE/train", avg_t_mae, epoch)
        writer.add_scalar("MAE/val", avg_v_mae, epoch)
        writer.add_scalar("MSE/train", avg_t_mse, epoch)
        writer.add_scalar("MSE/val", avg_v_mse, epoch)
        writer.add_scalar("RMSE/train", avg_t_rmse, epoch)
        writer.add_scalar("RMSE/val", avg_v_rmse, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        score_val = val_m[args.metric_for_best]
        if score_val < best_val_metric:
            best_val_metric = score_val
            torch.save(
                state_dict_for_save(model),
                os.path.join(args.ckpt_dir, "best_model.pt"),
            )

        if scheduler is not None and args.lr_scheduler != "onecycle":
            if args.lr_scheduler == "reduce_on_plateau":
                scheduler.step(score_val)
            else:
                scheduler.step()

        if args.dry_run:
            break

    writer.close()


if __name__ == "__main__":
    main()
