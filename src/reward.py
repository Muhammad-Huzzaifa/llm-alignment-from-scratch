from functools import partial
from pathlib import Path
from src.data import load_preference_splits
from src.batching import collate_pref
from src.models import load_causal_lm
from src.common import last_real_index
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import time
from datetime import timedelta


class RewardModel(nn.Module):

    def __init__(self, model_name_or_path):
        """load the sft policy as the backbone and add a linear reward head

        Args:
            model_name_or_path (str): path to checkpoints/sft/ (or a model name)
        """
        super().__init__()
        self.lm = load_causal_lm(model_name_or_path)  # load the causal LM model
        self.backbone = self.lm.model  # transformer backbone with no head
        self.reward_head = nn.Linear(self.lm.config.hidden_size, 1, bias=False, dtype=self.lm.dtype)  # no bias for reward head

    def forward(self, input_ids, attention_mask):
        """score a prompt plus summary with one scalar read from the last real token

        Args:
            input_ids (torch.Tensor): tensor of shape (batch, seq_len) with token ids
            attention_mask (torch.Tensor): tensor of shape (batch, seq_len) with 1s for real tokens

        Returns:
            torch.Tensor: rewards of shape (batch, 1)
        """
        hiddens = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # (batch, seq_len, hidden_size)
        idxs = last_real_index(attention_mask)  # (batch,) having the indices within the sequence
        batch = torch.arange(hiddens.size(0), device=hiddens.device)
        rewards = self.reward_head(hiddens[batch, idxs])  # (batch, 1)
        return rewards

    def save(self, chkpnt_dir):
        """save the causal LM and the reward head weights separately

        Args:
            chkpnt_dir (str): directory to save into
        """
        Path(chkpnt_dir).mkdir(parents=True, exist_ok=True)
        self.lm.save_pretrained(chkpnt_dir)  # save the underlying LM model
        torch.save(self.reward_head.state_dict(), f"{chkpnt_dir}/reward_head.pt")  # save the reward head weights

    @classmethod
    def load(cls, chkpnt_dir, device):
        """rebuild a frozen reward model from a saved checkpoint

        Args:
            chkpnt_dir (str): directory holding the saved LM and reward_head.pt
            device (str): device to place the model on

        Returns:
            RewardModel: the reward model in eval mode with every parameter frozen
        """
        m = cls.__new__(cls)
        nn.Module.__init__(m)
        m.lm = load_causal_lm(chkpnt_dir, device)
        m.backbone = m.lm.model
        m.reward_head = nn.Linear(m.lm.config.hidden_size, 1, bias=False, dtype=m.lm.dtype)
        m.reward_head.load_state_dict(torch.load(f"{chkpnt_dir}/reward_head.pt", map_location=device))
        m.to(device).eval()
        for p in m.parameters():
            p.requires_grad = False
        return m


def bt_loss(model, batch, device):
    """bradley-terry loss on a batch of preference pairs (Eq. 40)

    Args:
        model (RewardModel): reward model to train
        batch (dict): collate_pref output with the chosen and rejected ids and masks
        device (str): device to move the batch to

    Returns:
        torch.Tensor: scalar loss, -mean(logsigmoid(r_phi(x, y_w) - r_phi(x, y_l)))
    """
    chosen_ids = batch["chosen_ids"].to(device)
    chosen_mask = batch["chosen_mask"].to(device)
    rejected_ids = batch["rejected_ids"].to(device)
    rejected_mask = batch["rejected_mask"].to(device)

    ch_rewards = model(chosen_ids, chosen_mask)  # (batch, 1)
    rej_rewards = model(rejected_ids, rejected_mask)  # (batch, 1)

    loss = -torch.mean(
        torch.nn.functional.logsigmoid(ch_rewards - rej_rewards).float()  # (batch, 1)
    )

    return loss


def pairwise_accuracy(model, dataloader, device):
    """fraction of pairs where the chosen summary scores higher,
    a tie counts as half a win

    Args:
        model (RewardModel): reward model to evaluate
        dataloader (DataLoader): batches built with collate_pref
        device (str): device to move the batches to

    Returns:
        float: pairwise accuracy between 0 and 1
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

            ch_rewards = model(chosen_ids, chosen_mask)  # (batch, 1)
            rej_rewards = model(rejected_ids, rejected_mask)  # (batch, 1)

            correct += ((ch_rewards > rej_rewards).sum().item() + 0.5 * (ch_rewards == rej_rewards).sum().item())
            total += ch_rewards.size(0)

    return correct / total if total > 0 else 0


def train_reward(tok, batch_size, model_name_or_path, lr, grad_clip, chkpnt_dir, fig_dir, device):
    """train the reward model on the preference pairs for 1 epoch,
    logging the loss and validation pairwise accuracy every 250 steps

    Args:
        tok (AutoTokenizer): tokenizer used for padding and saving
        batch_size (int): number of preference pairs per step
        model_name_or_path (str): path to checkpoints/sft/, the starting backbone
        lr (float): AdamW learning rate
        grad_clip (float): gradient norm clipping value
        chkpnt_dir (str): directory to save the final reward model into
        fig_dir (str): directory to save the curves into
        device (str): device to train on

    Returns:
        dict: final train loss, final pairwise accuracy and elapsed time
    """
    train_ds, val_ds = load_preference_splits(tok)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=partial(collate_pref, pad_id=tok.pad_token_id))
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=partial(collate_pref, pad_id=tok.pad_token_id))

    model = RewardModel(model_name_or_path).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    train_losses, pairwise_accs = [], []  # record every 250th step
    run_loss, run_n = 0.0, 0
    prog_bar = tqdm(train_dl, desc="Training Reward")

    start_time = time.time()
    for idx, batch in enumerate(prog_bar):
        opt.zero_grad(set_to_none=True)
        loss = bt_loss(model, batch, device)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        run_loss += loss.item(); run_n += 1
        prog_bar.set_postfix({"train loss": f"{loss.item():.4f}"})

        if (idx + 1) % 250 == 0:
            train_loss = run_loss / run_n
            train_losses.append(train_loss); run_loss, run_n = 0.0, 0
            pairwise_acc = pairwise_accuracy(model, val_dl, device)
            pairwise_accs.append(pairwise_acc)
            print(f"step {idx + 1}/{len(train_dl)} - train loss: {train_loss:.4f} - pairwise accuracy: {pairwise_acc:.4f}")
            model.train()
    end_time = time.time()
    elapsed = str(timedelta(seconds=int(end_time - start_time)))

    Path(chkpnt_dir).mkdir(parents=True, exist_ok=True)
    model.save(chkpnt_dir)
    tok.save_pretrained(chkpnt_dir)

    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    _, (a1, a2) = plt.subplots(1, 2)
    a1.plot(train_losses, label="train loss")
    a1.set_xlabel("steps (x250)")
    a1.set_title("Training Bradley-Terry Loss")
    a2.plot(pairwise_accs, label="pairwise accuracy")
    a2.set_xlabel("steps (x250)")
    a2.set_title("Validation Pairwise Accuracy")
    plt.suptitle("Reward Model Training and Validation")
    plt.savefig(f"{fig_dir}/rwd_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "train_loss": train_losses[-1] if train_losses else 0.0,
        "pairwise_acc": pairwise_accs[-1] if pairwise_accs else 0.0,
        "elapsed_time": elapsed,
    }