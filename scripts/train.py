"""Task 3.2 - Train the encoder-decoder and report the loss curves.

Examples
--------
python scripts/train.py                                  # Transformer, defaults
python scripts/train.py --arch lstm_attn --epochs 20     # LSTM + Bahdanau attention
python scripts/train.py --epochs 2 --limit-batches 20    # smoke test
"""

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C
from data import make_loaders, pick_device
from model import build_model, count_parameters
from vocab import Vocabulary


def build_scheduler(optimizer, warmup_steps, total_steps):
    """Linear warm-up followed by cosine decay to 10% of the peak LR."""
    def lr_lambda(step):
        step = max(step, 1)
        if step < warmup_steps:
            return step / float(warmup_steps)
        progress = (step - warmup_steps) / max(1.0, total_steps - warmup_steps)
        progress = min(1.0, progress)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None,
              clip=C.CLIP_NORM, limit_batches=None, log_every=50, tag=""):
    """One pass over ``loader``. Returns (mean token loss, tokens seen)."""
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, total_tokens = 0.0, 0
    started = time.time()

    for step, (src, tgt_in, tgt_out) in enumerate(loader, start=1):
        if limit_batches and step > limit_batches:
            break
        src = src.to(device)
        tgt_in = tgt_in.to(device)
        tgt_out = tgt_out.to(device)

        with torch.set_grad_enabled(training):
            logits = model(src, tgt_in)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

        n_tokens = int((tgt_out != C.PAD_IDX).sum())
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        total_loss += float(loss.detach()) * n_tokens
        total_tokens += n_tokens

        if training and step % log_every == 0:
            elapsed = time.time() - started
            print("    {} step {:>4}/{}  loss {:.4f}  {:.0f} tok/s".format(
                tag, step, len(loader), total_loss / max(total_tokens, 1),
                total_tokens / max(elapsed, 1e-6)), flush=True)

    return total_loss / max(total_tokens, 1), total_tokens


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device, limit_batches=None):
    """Validation loss under an *unsmoothed* criterion (so exp(loss)=perplexity)."""
    return run_epoch(model, loader, criterion, device, limit_batches=limit_batches)[0]


def plot_history(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, [h["train_loss"] for h in history], "o-", label="train")
    axes[0].plot(epochs, [h["valid_loss"] for h in history], "s-", label="validation")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("cross-entropy per token")
    axes[0].set_title("Training / validation loss"); axes[0].grid(alpha=.3); axes[0].legend()

    axes[1].plot(epochs, [h["valid_ppl"] for h in history], "s-", color="crimson")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("perplexity")
    axes[1].set_title("Validation perplexity"); axes[1].grid(alpha=.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("Loss curves -> {}".format(path))


def main():
    p = argparse.ArgumentParser(description="Train the support response generator")
    p.add_argument("--arch", default=C.ARCH, choices=["transformer", "lstm_attn"])
    p.add_argument("--epochs", type=int, default=C.EPOCHS)
    p.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=C.LEARNING_RATE)
    p.add_argument("--patience", type=int, default=C.PATIENCE)
    p.add_argument("--device", default="auto")
    p.add_argument("--limit-batches", type=int, default=None)
    p.add_argument("--out", default=None, help="checkpoint path")
    args = p.parse_args()

    torch.manual_seed(C.RANDOM_SEED)
    os.makedirs(C.MODEL_DIR, exist_ok=True)
    os.makedirs(C.REPORT_DIR, exist_ok=True)

    device = pick_device(args.device)
    vocab = Vocabulary.load(C.VOCAB_JSON)
    loaders = make_loaders(vocab, batch_size=args.batch_size)

    model = build_model(args.arch, len(vocab), C).to(device)
    ckpt_path = args.out or (C.CHECKPOINT if args.arch == C.ARCH else
                             os.path.join(C.MODEL_DIR, "best_model_{}.pt".format(args.arch)))

    train_config = {
        "arch": args.arch, "vocab_size": len(vocab), "device": str(device),
        "batch_size": args.batch_size, "epochs": args.epochs,
        "lr": args.lr, "weight_decay": C.WEIGHT_DECAY,
        "label_smoothing": C.LABEL_SMOOTHING, "clip_norm": C.CLIP_NORM,
        "warmup_steps": C.WARMUP_STEPS, "patience": args.patience,
        "max_src_len": C.MAX_SRC_LEN, "max_tgt_len": C.MAX_TGT_LEN,
        "parameters": count_parameters(model),
        "train_pairs": len(loaders["train"].dataset),
        "valid_pairs": len(loaders["valid"].dataset),
        "test_pairs": len(loaders["test"].dataset),
    }
    if args.arch == "transformer":
        train_config.update({"d_model": C.D_MODEL, "n_heads": C.N_HEADS,
                             "enc_layers": C.N_ENC_LAYERS, "dec_layers": C.N_DEC_LAYERS,
                             "ffn_dim": C.FFN_DIM, "dropout": C.DROPOUT})
    else:
        train_config.update({"emb_dim": C.EMB_DIM, "hidden_dim": C.HIDDEN_DIM,
                             "lstm_layers": C.LSTM_LAYERS, "dropout": C.DROPOUT,
                             "attention": "bahdanau (additive)"})

    print("=" * 78)
    print("Training configuration")
    for k, v in train_config.items():
        print("  {:<18} {}".format(k, v))
    print("=" * 78)

    # Smoothed loss for optimisation, plain CE for reporting perplexity.
    train_criterion = nn.CrossEntropyLoss(ignore_index=C.PAD_IDX,
                                          label_smoothing=C.LABEL_SMOOTHING)
    eval_criterion = nn.CrossEntropyLoss(ignore_index=C.PAD_IDX)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=C.WEIGHT_DECAY, betas=(0.9, 0.98))
    steps_per_epoch = args.limit_batches or len(loaders["train"])
    scheduler = build_scheduler(optimizer, C.WARMUP_STEPS,
                                steps_per_epoch * args.epochs)

    history, best_valid, bad_epochs = [], float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, _ = run_epoch(model, loaders["train"], train_criterion, device,
                                  optimizer=optimizer, scheduler=scheduler,
                                  limit_batches=args.limit_batches,
                                  tag="e{}".format(epoch))
        valid_loss = evaluate_loss(model, loaders["valid"], eval_criterion, device,
                                   limit_batches=args.limit_batches)
        valid_ppl = math.exp(min(valid_loss, 20))
        history.append({"epoch": epoch, "train_loss": round(train_loss, 4),
                        "valid_loss": round(valid_loss, 4),
                        "valid_ppl": round(valid_ppl, 3),
                        "lr": round(scheduler.get_last_lr()[0], 6),
                        "seconds": round(time.time() - t0, 1)})
        print("epoch {:>2}/{}  train {:.4f}  valid {:.4f}  ppl {:.2f}  ({:.0f}s)".format(
            epoch, args.epochs, train_loss, valid_loss, valid_ppl,
            time.time() - t0), flush=True)

        if valid_loss < best_valid - 1e-4:
            best_valid, bad_epochs = valid_loss, 0
            torch.save({"model_state": model.state_dict(),
                        "config": train_config,
                        "epoch": epoch,
                        "valid_loss": valid_loss}, ckpt_path)
            print("           new best -> {}".format(os.path.relpath(ckpt_path, C.ROOT_DIR)))
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("Early stopping: validation loss has not improved for "
                      "{} epochs.".format(args.patience))
                break

        with open(C.HISTORY_JSON, "w") as fh:
            json.dump({"config": train_config, "history": history}, fh, indent=2)

    with open(C.HISTORY_JSON, "w") as fh:
        json.dump({"config": train_config, "history": history}, fh, indent=2)
    plot_history(history, C.LOSS_CURVE_PNG)

    print("\nBest validation loss {:.4f} (perplexity {:.2f})".format(
        best_valid, math.exp(min(best_valid, 20))))
    print("Checkpoint: {}".format(ckpt_path))


if __name__ == "__main__":
    main()
