from functools import partial
from pathlib import Path
from src.data import load_preference_splits
from src.batching import collate_pref
from src.models import load_causal_lm, load_frozen
from src.common import sequence_logprobs
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import time
from datetime import timedelta


def dpo_loss(model, ref_model, batch, beta, device):
    """dpo loss on a batch of preference pairs, the policy is its own implicit
    reward and the reference only sets the offset (Eq. 60)

    Args:
        model (nn.Module): policy pi_theta being trained
        ref_model (nn.Module): frozen reference pi_ref
        batch (dict): collate_pref output with the chosen and rejected ids and masks
        beta (float): temperature of the implicit reward
        device (str): device to move the batch to

    Returns:
        torch.Tensor: scalar loss, -mean(logsigmoid(shift))
    """
    chosen_ids = batch["chosen_ids"].to(device)
    chosen_mask = batch["chosen_mask"].to(device)
    rejected_ids = batch["rejected_ids"].to(device)
    rejected_mask = batch["rejected_mask"].to(device)
    prompt_lens = batch["prompt_lens"].to(device)

    chosen = sequence_logprobs(model, chosen_ids, chosen_mask, prompt_lens)
    rejected = sequence_logprobs(model, rejected_ids, rejected_mask, prompt_lens)

    chosen_ref = sequence_logprobs(ref_model, chosen_ids, chosen_mask, prompt_lens)
    rejected_ref = sequence_logprobs(ref_model, rejected_ids, rejected_mask, prompt_lens)

    # compute the DPO loss
    loss = -torch.mean(
        torch.nn.functional.logsigmoid(
            beta * ((chosen - chosen_ref) - (rejected - rejected_ref))
        )
    )

    return loss


def implicit_accuracy(model, ref_model, dataloader, device):
    """fraction of pairs where the implicit reward prefers the chosen summary,
    a tie counts as half a win

    Args:
        model (nn.Module): policy pi_theta
        ref_model (nn.Module): frozen reference pi_ref
        dataloader (DataLoader): batches built with collate_pref
        device (str): device to move the batches to

    Returns:
        float: implicit reward accuracy between 0 and 1
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            chosen_ids = batch["chosen_ids"].to(device)
            chosen_mask = batch["chosen_mask"].to(device)
            rejected_ids = batch["rejected_ids"].to(device)
            rejected_mask = batch["rejected_mask"].to(device)
            prompt_lens = batch["prompt_lens"].to(device)

            chosen = sequence_logprobs(model, chosen_ids, chosen_mask, prompt_lens)
            rejected = sequence_logprobs(model, rejected_ids, rejected_mask, prompt_lens)

            chosen_ref = sequence_logprobs(ref_model, chosen_ids, chosen_mask, prompt_lens)
            rejected_ref = sequence_logprobs(ref_model, rejected_ids, rejected_mask, prompt_lens)

            # compute the implicit reward accuracy
            z = (chosen - chosen_ref) - (rejected - rejected_ref)
            correct += (z > 0).sum().item() + 0.5 * (z == 0).sum().item()  # ties are treated as 50% accuracy which is enevtually coin toss
            total += z.size(0)

    return correct / total if total > 0 else 0


def train_dpo(tok, batch_size, model_name_or_path, lr, grad_clip, beta, chkpnt_dir, fig_dir, device):
    """train dpo on the preference pairs for 1 epoch,
    logging the loss and validation implicit reward accuracy every 250 steps

    Args:
        tok (AutoTokenizer): tokenizer used for padding and saving
        batch_size (int): number of preference pairs per step
        model_name_or_path (str): path to checkpoints/sft/, used for the policy and the reference
        lr (float): AdamW learning rate
        grad_clip (float): gradient norm clipping value
        beta (float): temperature of the implicit reward
        chkpnt_dir (str): directory to save the final policy into
        fig_dir (str): directory to save the curves into
        device (str): device to train on

    Returns:
        dict: final train loss, final implicit reward accuracy and elapsed time
    """
    train_ds, val_ds = load_preference_splits(tok)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=partial(collate_pref, pad_id=tok.pad_token_id))
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=partial(collate_pref, pad_id=tok.pad_token_id))

    model = load_causal_lm(model_name_or_path, device); model.train()
    ref_model = load_frozen(model_name_or_path, device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    train_losses, imp_rwd_accs = [], []  # record every 250th step
    run_loss, run_n = 0.0, 0
    prog_bar = tqdm(train_dl, desc="Training DPO")

    start_time = time.time()
    for idx, batch in enumerate(prog_bar):
        opt.zero_grad(set_to_none=True)
        loss = dpo_loss(model, ref_model, batch, beta, device)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        run_loss += loss.item(); run_n += 1
        prog_bar.set_postfix({"train loss": f"{loss.item():.4f}"})

        if (idx + 1) % 250 == 0:
            train_loss = run_loss / run_n
            train_losses.append(train_loss); run_loss, run_n = 0.0, 0
            imp_rwd_acc = implicit_accuracy(model, ref_model, val_dl, device)
            imp_rwd_accs.append(imp_rwd_acc)
            print(f"step {idx + 1}/{len(train_dl)} - train loss: {train_loss:.4f} - implicit reward accuracy: {imp_rwd_acc:.4f}")
            model.train()
    end_time = time.time()
    elapsed = str(timedelta(seconds=int(end_time - start_time)))

    Path(chkpnt_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(chkpnt_dir)
    tok.save_pretrained(chkpnt_dir)

    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    _, (a1, a2) = plt.subplots(1, 2)
    a1.plot(train_losses, label="train loss")
    a1.set_xlabel("steps (x250)")
    a1.set_title("Training DPO Loss")
    a2.plot(imp_rwd_accs, label="implicit reward accuracy")
    a2.set_xlabel("steps (x250)")
    a2.set_title("Validation Implicit Reward Accuracy")
    plt.suptitle("DPO Training and Validation")
    plt.savefig(f"{fig_dir}/dpo_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "train_loss": train_losses[-1] if train_losses else 0.0,
        "imp_rwd_acc": imp_rwd_accs[-1] if imp_rwd_accs else 0.0,
        "elapsed_time": elapsed,
    }