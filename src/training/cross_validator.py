"""
cross_validator.py — Stratified K-Fold CV untuk uji stabilitas model.
Pool = train + val. Test set TIDAK disentuh sama sekali.
"""
import logging
from copy import deepcopy
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import DataLoader
from src.data.dataset_loader import FaceDataset
from src.models.backbone import FaceRecognitionModel
from src.training.trainer import Trainer

log = logging.getLogger(__name__)

def run_cv(pool_X, pool_y, num_classes, cfg, train_tf, val_tf, device) -> dict:
    n_splits = cfg['cross_validation']['n_splits']
    seed     = cfg['project']['seed']
    skf      = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X, y     = np.array(pool_X), np.array(pool_y)
    fold_accs = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        log.info(f"Fold {fold+1}/{n_splits}")
        fold_cfg = deepcopy(cfg)
        fold_cfg['training']['phase1_epochs'] = 3
        fold_cfg['training']['phase2_epochs'] = 7
        fold_cfg['training']['patience']      = 5
        fold_cfg['paths']['checkpoints']      = cfg['paths']['checkpoints'] + f'/cv_fold_{fold+1}'

        bs = cfg['training']['batch_size']
        nw = cfg['data']['num_workers']
        tr_loader  = DataLoader(FaceDataset(X[tr_idx].tolist(), y[tr_idx].tolist(), train_tf),
                                batch_size=bs, shuffle=True, num_workers=nw, drop_last=True)
        val_loader = DataLoader(FaceDataset(X[val_idx].tolist(), y[val_idx].tolist(), val_tf),
                                batch_size=bs, shuffle=False, num_workers=nw)

        model   = FaceRecognitionModel(num_classes, fold_cfg).to(device)
        trainer = Trainer(model, tr_loader, val_loader, fold_cfg, device)
        history = trainer.train()
        fold_accs.append(history.best_val_acc())
        log.info(f"Fold {fold+1} best: {fold_accs[-1]:.4f}")

    mean_acc = float(np.mean(fold_accs))
    std_acc  = float(np.std(fold_accs))
    log.info(f"CV: {mean_acc:.4f} +/- {std_acc:.4f}")
    return {'fold_accs': fold_accs, 'mean_acc': mean_acc, 'std_acc': std_acc}
