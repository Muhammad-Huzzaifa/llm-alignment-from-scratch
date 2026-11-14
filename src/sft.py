from functools import partial
from pathlib import Path
from src.data import load_sft_splits
from src.batching import collate_sft
from src.models import load_causal_lm
from src.common import token_logprobs
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import time
from datetime import timedelta


def sft_loss(model, batch, device):
    """mean negative log-likelihood of the summary tokens only,
    prompt tokens and padding contribute nothing

    Args:
        model (nn.Module): causal LM being fine-tuned
        batch (dict): collate_sft output with "input_ids", "attention_mask" and "prompt_lens"
        device (str): device to move the batch to

    Returns:
        torch.Tensor: scalar loss, the mean nll over the summary tokens
    """
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    prompt_lens = batch["prompt_lens"].to(device)

    tok_logprobs, summary_mask = token_logprobs(model, input_ids, attention_mask, prompt_lens)

    return -tok_logprobs[summary_mask].mean()  # mean nll over summary tokens


def evaluate_sft(model, dataloader, device):
    """token-weighted mean nll of the summary tokens over a whole dataloader

    Args:
        model (nn.Module): causal LM to evaluate
        dataloader (DataLoader): batches built with collate_sft
        device (str): device to move the batches to

    Returns:
        float: mean nll over all summary tokens, inf if the loader is empty
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            prompt_lens = batch["prompt_lens"].to(device)

            tok_logprobs, summary_mask = token_logprobs(model, input_ids, attention_mask, prompt_lens)

            total_loss += -tok_logprobs[summary_mask].sum().item()  # sum nll over summary tokens
            total_tokens += summary_mask.sum().item()

    return total_loss / total_tokens if total_tokens > 0 else float('inf')  # mean nll over summary tokens


def train_sft(tok, batch_size, model_name_or_path, lr, grad_clip, chkpnt_dir, fig_dir, device):
    """supervised fine-tuning on the sft pairs for 1 epoch,
    logging train and validation loss every 250 steps

    Args:
        tok (AutoTokenizer): tokenizer used for padding and saving
        batch_size (int): number of sft pairs per step
        model_name_or_path (str): base model name or path to start from
        lr (float): AdamW learning rate
        grad_clip (float): gradient norm clipping value
        chkpnt_dir (str): directory to save the final model into
        fig_dir (str): directory to save the loss curves into
        device (str): device to train on

    Returns:
        dict: final train loss, final val loss and elapsed time
    """
    train_ds, val_ds = load_sft_splits(tok)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=partial(collate_sft, pad_id=tok.pad_token_id))
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=partial(collate_sft, pad_id=tok.pad_token_id))

    model = load_causal_lm(model_name_or_path, device); model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    
    train_losses, val_losses = [], []  # record every 250th step
    run_loss, run_n = 0.0, 0
    prog_bar = tqdm(train_dl, desc="Training SFT")

    start_time = time.time()
    for idx, batch in enumerate(prog_bar):
        opt.zero_grad(set_to_none=True)
        loss = sft_loss(model, batch, device)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        run_loss += loss.item(); run_n += 1
        prog_bar.set_postfix({"train loss": f"{loss.item():.4f}"})

        if (idx + 1) % 250 == 0:
            train_loss = run_loss / run_n
            train_losses.append(train_loss); run_loss, run_n = 0.0, 0
            val_loss = evaluate_sft(model, val_dl, device)
            val_losses.append(val_loss)
            print(f"step {idx + 1}/{len(train_dl)} - train loss: {train_loss:.4f} - val loss: {val_loss:.4f}")
            model.train()
    end_time = time.time()
    elapsed = str(timedelta(seconds=int(end_time - start_time)))

    Path(chkpnt_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(chkpnt_dir)
    tok.save_pretrained(chkpnt_dir)

    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    plt.plot(train_losses, label="train loss")
    plt.plot(val_losses, label="val loss")
    plt.xlabel("steps (x250)")
    plt.ylabel("mean nll over summary tokens")
    plt.title("SFT Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{fig_dir}/sft_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "train_loss": train_losses[-1] if train_losses else 0,
        "val_loss": val_losses[-1] if val_losses else 0,
        "elapsed_time": elapsed,
    }