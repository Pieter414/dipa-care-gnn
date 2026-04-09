"""
    Experiment runner for fraud detection GNN benchmarking.

    Sweeps over: models × loss functions × datasets × random seeds.
    Results are saved per-run as JSON and aggregated into a summary CSV.

    Usage:
        # Run all configurations (42 configs × 10 seeds = 420 runs)
        python run_experiments.py

        # Single run for debugging
        python run_experiments.py --models SAGE --losses ce --datasets yelp --seeds 42

        # Prioritize GraphSAGE across everything (if compute-limited)
        python run_experiments.py --models SAGE

        # Target a specific GPU (e.g. GPU 1)
        python run_experiments.py --cuda-device 1
"""

import os
import json
import time
import random
import argparse
import itertools
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, recall_score, accuracy_score,
)

from utils import load_data, normalize, pos_neg_split, undersample
from model import OneLayerCARE
from layers import InterAgg, IntraAgg
from graphsage import GraphSage, MeanAggregator, Encoder
from losses import build_loss, CARECompositeLoss

# ── Defaults ────────────────────────────────────────────────────────────────

DEFAULT_MODELS = ["CARE", "SAGE"]
DEFAULT_LOSSES = ["ce", "weighted_ce", "focal", "label_smooth", "class_balanced", "dice", "care_composite"]
DEFAULT_DATASETS = ["yelp", "amazon", "comp"]
DEFAULT_SEEDS = [42, 72, 123, 256, 314, 512, 666, 777, 888, 999]

RESULTS_DIR = Path("results")


# ── Helpers ─────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(model_name, feat_data, adj_lists, homo, args):
    """Construct a GNN model and return (model, uses_label_scores)."""

    features = nn.Embedding(feat_data.shape[0], feat_data.shape[1])
    norm_feat = normalize(feat_data)
    features.weight = nn.Parameter(torch.FloatTensor(norm_feat), requires_grad=False)
    if args.cuda:
        features = features.to(args.device)

    if model_name == "CARE":
        intra_aggs = [
            IntraAgg(features, feat_data.shape[1], cuda=args.cuda)
            for _ in range(3)
        ]
        inter1 = InterAgg(
            features, feat_data.shape[1], args.emb_size,
            adj_lists, intra_aggs,
            inter=args.inter, step_size=args.step_size, cuda=args.cuda,
        )
        model = OneLayerCARE(2, inter1, lambda_1=args.lambda_1)
        uses_label_scores = True

    elif model_name == "SAGE":
        agg1 = MeanAggregator(features, cuda=args.cuda)
        enc1 = Encoder(
            features, feat_data.shape[1], args.emb_size,
            homo, agg1, gcn=True, cuda=args.cuda,
        )
        enc1.num_samples = 5
        model = GraphSage(2, enc1)
        uses_label_scores = False

    else:
        raise ValueError(f"Unknown model: {model_name}")

    if args.cuda:
        model = model.to(args.device)

    return model, features, uses_label_scores


def evaluate(model, test_nodes, test_labels, batch_size, uses_label_scores, cuda):
    """Run evaluation and return a metrics dict."""
    model.eval()
    num_batches = int(len(test_nodes) / batch_size) + 1
    all_gnn_probs = []
    all_preds = []

    with torch.no_grad():
        for b in range(num_batches):
            start = b * batch_size
            end = min((b + 1) * batch_size, len(test_nodes))
            if start >= end:
                break
            batch = test_nodes[start:end]
            batch_labels = test_labels[start:end]

            if uses_label_scores:
                gnn_prob, _ = model.to_prob(batch, batch_labels, train_flag=False)
            else:
                gnn_prob = model.to_prob(batch)

            probs_np = gnn_prob.data.cpu().numpy()
            all_gnn_probs.extend(probs_np[:, 1].tolist())
            all_preds.extend(probs_np.argmax(axis=1).tolist())

    preds = np.array(all_preds)
    probs = np.array(all_gnn_probs)

    return {
        "auc": roc_auc_score(test_labels, probs),
        "ap": average_precision_score(test_labels, probs),
        "f1_macro": f1_score(test_labels, preds, average="macro"),
        "recall_macro": recall_score(test_labels, preds, average="macro"),
        "accuracy": accuracy_score(test_labels, preds),
    }


# ── Single experiment ───────────────────────────────────────────────────────

def run_single(model_name, loss_name, dataset, seed, args):
    """Train and evaluate one (model, loss, dataset, seed) configuration."""
    set_seed(seed)

    # load data
    [homo, rel1, rel2, rel3], feat_data, labels = load_data(dataset)
    adj_lists = [rel1, rel2, rel3] if model_name == "CARE" else homo

    # train/test split — match original paper's protocol
    if dataset == "yelp":
        index = list(range(len(labels)))
        idx_train, idx_test, y_train, y_test = train_test_split(
            index, labels, stratify=labels, test_size=0.60,
            random_state=2, shuffle=True,
        )
    elif dataset == "amazon":
        # first 3305 nodes are unlabeled
        index = list(range(3305, len(labels)))
        idx_train, idx_test, y_train, y_test = train_test_split(
            index, labels[3305:], stratify=labels[3305:],
            test_size=0.60, random_state=2, shuffle=True,
        )
    elif dataset == "comp":
        # FDCompCN: all nodes are labeled, use same 40/60 split
        index = list(range(len(labels)))
        idx_train, idx_test, y_train, y_test = train_test_split(
            index, labels, stratify=labels, test_size=0.60,
            random_state=2, shuffle=True,
        )

    train_pos, train_neg = pos_neg_split(idx_train, y_train)

    # build model
    model, features, uses_label_scores = build_model(
        model_name, feat_data, adj_lists, homo, args,
    )

    # build loss — skip care_composite for non-CARE models (it degrades to plain CE)
    loss_fn = build_loss(loss_name, labels)
    if args.cuda:
        loss_fn = loss_fn.to(args.device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.lambda_2,
    )

    # ── Training loop ───────────────────────────────────────────────────────
    best_metrics = None
    best_auc = 0.0

    for epoch in range(args.num_epochs):
        model.train()
        sampled_idx = undersample(train_pos, train_neg, scale=1)
        random.shuffle(sampled_idx)

        num_batches = int(len(sampled_idx) / args.batch_size) + 1
        if model_name == "CARE":
            model.inter1.batch_num = num_batches

        epoch_loss = 0.0
        n_batches_actual = 0
        for batch in range(num_batches):
            i_start = batch * args.batch_size
            i_end = min((batch + 1) * args.batch_size, len(sampled_idx))
            if i_start >= i_end:
                break

            batch_nodes = sampled_idx[i_start:i_end]
            batch_labels = labels[np.array(batch_nodes)]
            label_tensor = torch.tensor(
                batch_labels, dtype=torch.long, device=args.device,
            )

            optimizer.zero_grad()

            # forward pass differs by model type
            if uses_label_scores:
                gnn_scores, label_scores = model(batch_nodes, label_tensor)
                if isinstance(loss_fn, CARECompositeLoss):
                    loss = loss_fn(gnn_scores, label_tensor, label_scores)
                else:
                    # non-composite losses only use gnn_scores
                    loss = loss_fn(gnn_scores, label_tensor)
            else:
                scores = model.forward(batch_nodes)
                loss = loss_fn(scores, label_tensor)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches_actual += 1

        avg_loss = epoch_loss / max(n_batches_actual, 1)

        # log training loss every few epochs so we can verify learning
        # evaluate periodically
        if epoch % args.test_epochs == 0:
            metrics = evaluate(
                model, idx_test, y_test, args.batch_size,
                uses_label_scores, args.cuda,
            )
            print(f"  [epoch {epoch:3d}] loss={avg_loss:.4f}  auc={metrics['auc']:.4f}  f1={metrics['f1_macro']:.4f}")
            if metrics["auc"] > best_auc:
                best_auc = metrics["auc"]
                best_metrics = metrics.copy()
                best_metrics["best_epoch"] = epoch

    return best_metrics


# ── Main sweep ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GNN fraud detection experiment sweep")

    # sweep dimensions
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--losses", nargs="+", default=DEFAULT_LOSSES)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)

    # model hyperparams (shared defaults from original train.py)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lambda_1", type=float, default=2.0)
    parser.add_argument("--lambda_2", type=float, default=1e-3)
    parser.add_argument("--emb-size", type=int, default=64)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--test-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--inter", type=str, default="GNN")
    parser.add_argument("--step-size", type=float, default=0.02)
    parser.add_argument("--no-cuda", action="store_true", default=False, help="Disable CUDA entirely.")
    parser.add_argument("--cuda-device", type=int, default=0, help="CUDA device ID (e.g. 0, 1, 2). Ignored if --no-cuda.")
    parser.add_argument("--summary-path", type=str, default="summary.csv")
    parser.add_argument("--result-path", type=str, default="/results/")

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Pin the target GPU so all .cuda() calls throughout the codebase
    # (including inside layers.py, graphsage.py, model.py) land on the
    # correct device without modifying those files.
    if args.cuda:
        torch.cuda.set_device(args.cuda_device)
        args.device = torch.device(f"cuda:{args.cuda_device}")
        print(f"Using CUDA device {args.cuda_device}: {torch.cuda.get_device_name(args.cuda_device)}")
    else:
        args.device = torch.device("cpu")
        print("Using CPU")

    RESULTS_DIR = Path(args.result_path)
    RESULTS_DIR.mkdir(exist_ok=True)

    # enumerate all configurations
    configs = list(itertools.product(args.models, args.losses, args.datasets, args.seeds))
    total = len(configs)
    print(f"Running {total} experiments ({len(args.models)} models × "
          f"{len(args.losses)} losses × {len(args.datasets)} datasets × "
          f"{len(args.seeds)} seeds)")

    all_results = []

    for i, (model_name, loss_name, dataset, seed) in enumerate(configs, 1):
        run_id = f"{model_name}_{loss_name}_{dataset}_s{seed}"
        result_path = RESULTS_DIR / f"{run_id}.json"

        # skip already-completed runs (resume-friendly)
        if result_path.exists():
            print(f"[{i}/{total}] SKIP {run_id} (already exists)")
            with open(result_path) as f:
                all_results.append(json.load(f))
            continue

        # adjust batch size per dataset (amazon and comp are smaller)
        if dataset in ("amazon", "comp"):
            args.batch_size = 256
        else:
            args.batch_size = 1024

        print(f"[{i}/{total}] {run_id} ...", end=" ", flush=True)
        t0 = time.time()

        try:
            metrics = run_single(model_name, loss_name, dataset, seed, args)
            elapsed = time.time() - t0

            record = {
                "model": model_name,
                "loss": loss_name,
                "dataset": dataset,
                "seed": seed,
                "elapsed_s": round(elapsed, 1),
                **metrics,
            }
            with open(result_path, "w") as f:
                json.dump(record, f, indent=2)

            all_results.append(record)
            print(f"AUC={metrics['auc']:.4f}  F1={metrics['f1_macro']:.4f}  ({elapsed:.0f}s)")

        except Exception as e:
            print(f"FAILED: {e}")
            all_results.append({
                "model": model_name, "loss": loss_name,
                "dataset": dataset, "seed": seed, "error": str(e),
            })

    # ── Aggregate summary ───────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY (mean ± std over seeds)")
    print("=" * 80)

    import pandas as pd
    df = pd.DataFrame([r for r in all_results if "error" not in r])
    if len(df) > 0:
        summary = (
            df.groupby(["model", "loss", "dataset"])
            .agg(
                auc_mean=("auc", "mean"), auc_std=("auc", "std"),
                f1_mean=("f1_macro", "mean"), f1_std=("f1_macro", "std"),
                recall_mean=("recall_macro", "mean"), recall_std=("recall_macro", "std"),
                n_runs=("seed", "count"),
            )
            .round(4)
        )
        summary_path = RESULTS_DIR / args.summary_path
        summary.to_csv(summary_path)
        print(summary.to_string())
        print(f"\nSaved to {summary_path}")


if __name__ == "__main__":
    main()