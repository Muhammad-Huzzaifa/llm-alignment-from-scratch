from pathlib import Path
from src.data import load_sft_splits
from src.batching import collate_prompts
from src.models import load_causal_lm, load_frozen
from src.reward import RewardModel
from src.common import generate_summaries
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import time
from datetime import timedelta
import random


class PolicyWithValue(nn.Module):

    def __init__(self, model_name_or_path):
        """build the policy from the sft checkpoint and add a fresh value head

        Args:
            model_name_or_path (str): path to checkpoints/sft/ (or a model name)
        """
        super().__init__()
        self.policy = load_causal_lm(model_name_or_path)
        self.value_head = nn.Linear(self.policy.config.hidden_size, 1, bias=False, dtype=self.policy.dtype)

    def forward(self, input_ids, attention_mask):
        """plain causal LM forward, logits only so the common helpers keep working

        Args:
            input_ids (torch.Tensor): tensor of shape (batch, seq_len) with token ids
            attention_mask (torch.Tensor): tensor of shape (batch, seq_len) with 1s for real tokens

        Returns:
            torch.Tensor: logits of shape (batch, seq_len, vocab)
        """
        return self.policy(input_ids=input_ids, attention_mask=attention_mask).logits

    def hidden(self, input_ids, attention_mask):
        """last hidden states of the transformer body, no lm_head applied

        Args:
            input_ids (torch.Tensor): tensor of shape (batch, seq_len) with token ids
            attention_mask (torch.Tensor): tensor of shape (batch, seq_len) with 1s for real tokens

        Returns:
            torch.Tensor: hidden states of shape (batch, seq_len, hidden_size)
        """
        return self.policy.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    def values(self, hidden):
        """value V(s_t) for every position from the value head

        Args:
            hidden (torch.Tensor): tensor of shape (batch, seq_len, hidden_size)

        Returns:
            torch.Tensor: values of shape (batch, seq_len)
        """
        return self.value_head(hidden).squeeze(-1)


def rollout(model, ref_model, rwd_model, prompts, tok, max_new_tokens, do_sample, temperature, top_k, top_p, device):
    """sample one summary per prompt and store everything the update needs,
    without building a graph

    Args:
        model (PolicyWithValue): policy pi_theta together with its value head
        ref_model (nn.Module): frozen reference pi_ref
        rwd_model (RewardModel): frozen reward model r_phi
        prompts (dict): collate_prompts output with left-padded "input_ids" and "attention_mask"
        tok (AutoTokenizer): tokenizer
        max_new_tokens (int): at most this many summary tokens per prompt
        do_sample (bool): sample the summaries instead of greedy decoding
        temperature (float): sampling temperature
        top_k (int): top-k sampling parameter, 0 disables it
        top_p (float): top-p sampling parameter, 1.0 disables it
        device (str): device to run the rollout on

    Returns:
        dict: full ids and mask, generated ids, keep mask, old and reference logprobs,
            old values, reward-model scores and the prompt length
    """
    prompt_ids = prompts["input_ids"].to(device)
    prompt_mask = prompts["attention_mask"].to(device)
    prompt_len = prompt_ids.size(1)
    
    model.eval()
    with torch.no_grad():
        new_ids = generate_summaries(model.policy, tok, prompt_ids, prompt_mask, max_new_tokens, do_sample, temperature, top_k, top_p)
        gen_len = new_ids.size(1)

        is_eos = (new_ids == tok.eos_token_id).long()  # [a, b, c, eos, d, e] -> [0, 0, 0, 1, 0, 0]
        keep = (is_eos.cumsum(dim=1) - is_eos) == 0  # [0, 0, 0, 1, 0, 0] -> [0, 0, 0, 1, 1, 1] - [0, 0, 0, 1, 0, 0] = [0, 0, 0, 0, 1, 1] -> [1, 1, 1, 1, 0, 0]

        full_ids = torch.cat([prompt_ids, new_ids], dim=1)  # (batch, prompt_len+gen_len)
        full_mask = torch.cat([prompt_mask, keep], dim=1)  # (batch, prompt_len+gen_len)

        action_slice = slice(prompt_len-1, prompt_len+gen_len-1)  # summary tokens indices - 1
        
        hidden = model.hidden(full_ids, full_mask)  # (batch, prompt_len+gen_len, hidden_size)
        old_logits = model.policy.lm_head(hidden[:, action_slice])  # (batch, gen_len, vocab)
        old_logp = -torch.nn.functional.cross_entropy(
            old_logits.reshape(-1, old_logits.size(-1)), new_ids.reshape(-1), reduction="none"
        ).view(new_ids.size(0), gen_len) # (batch, gen_len)

        values = model.values(hidden)[:, action_slice]  # (batch, gen_len)

        ref_hidden = ref_model.model(full_ids, full_mask).last_hidden_state  # (batch, prompt_len+gen_len, hidden_size)
        ref_logits = ref_model.lm_head(ref_hidden[:, action_slice])  # (batch, gen_len, vocab)
        ref_logp = -torch.nn.functional.cross_entropy(
            ref_logits.reshape(-1, ref_logits.size(-1)), new_ids.reshape(-1), reduction="none"
        ).view(new_ids.size(0), gen_len) # (batch, gen_len)

        reward_score = rwd_model(full_ids, full_mask).squeeze(-1)  # (batch,)

    return {
        "full_ids": full_ids.detach(),
        "full_mask": full_mask.detach(),
        "new_ids": new_ids.detach(),
        "keep": keep.detach(),
        "old_logp": old_logp.detach(),
        "ref_logp": ref_logp.detach(),
        "values": values.detach(),
        "reward_scores": reward_score.detach(),
        "prompt_len": prompt_len
    }


def per_token_reward(old_logp, ref_logp, reward_scores, keep, beta):
    """kl penalty at every generated token plus r_phi(x, y) at the last one (Eq. 41)

    Args:
        old_logp (torch.Tensor): tensor of shape (batch, gen_len) with log pi_theta_old(a_t | s_t)
        ref_logp (torch.Tensor): tensor of shape (batch, gen_len) with log pi_ref(a_t | s_t)
        reward_scores (torch.Tensor): tensor of shape (batch,) with r_phi(x, y) of each summary
        keep (torch.Tensor): tensor of shape (batch, gen_len) with 1 up to the first eos
        beta (float): beta_KL coefficient

    Returns:
        tuple: (rewards, kl) both of shape (batch, gen_len)
    """
    kl = old_logp - ref_logp  # (batch, gen_len)
    rewards = -beta * kl  # (batch, gen_len)

    last = keep.long().sum(dim=1) - 1  # index of the last valid token in each row (batch,)
    rewards = rewards.scatter_add(
        dim=1, index=last.unsqueeze(1), src=reward_scores.unsqueeze(1)  # add the reward at last time step to every reward
    )  # (batch, gen_len)
    rewards = rewards * keep

    return rewards, kl


def returns_and_advantages(rewards, values, keep, gamma):
    """discounted returns and normalised advantages (Eqs. 19, 20 and 27)

    Args:
        rewards (torch.Tensor): tensor of shape (batch, gen_len) with per-token rewards
        values (torch.Tensor): tensor of shape (batch, gen_len) with V(s_t)
        keep (torch.Tensor): tensor of shape (batch, gen_len) with 1 for real generated tokens
        gamma (float): discount factor, 1.0 for this assignment

    Returns:
        tuple: (returns, advantages) both of shape (batch, gen_len),
            advantages centred and scaled over all generated tokens
    """
    keep_f = keep.to(rewards.dtype)
    discounted = rewards * keep_f  # (batch, gen_len)

    returns = torch.zeros_like(discounted)  # (batch, gen_len)
    running = torch.zeros(discounted.size(0), device=discounted.device, dtype=discounted.dtype)  # (batch,)
    for t in reversed(range(discounted.size(1))):
        running = discounted[:, t] + gamma * running
        returns[:, t] = running
    returns = returns * keep_f

    advantages = returns - values  # (batch, gen_len)
    valid = advantages[keep]
    advantages = (advantages - valid.mean()) / (valid.std(unbiased=False) + 1e-8)
    advantages = advantages * keep_f

    return returns, advantages


def ppo_loss(model, roll, returns, advantages, clip_eps, c_v, c_h):
    """clipped policy loss, value loss and entropy bonus on one stored rollout (Eqs. 32, 34, 35)

    Args:
        model (PolicyWithValue): policy pi_theta being trained
        roll (dict): stored rollout returned by rollout()
        returns (torch.Tensor): tensor of shape (batch, gen_len) with R_t
        advantages (torch.Tensor): tensor of shape (batch, gen_len) with normalised A_t
        clip_eps (float): epsilon of the probability ratio clip
        c_v (float): value loss coefficient
        c_h (float): entropy coefficient

    Returns:
        tuple: (loss, stats) where stats holds the three loss terms, the clip fraction
            and the approximate kl between the old and the updated policy
    """
    new_ids = roll["new_ids"]
    keep = roll["keep"]
    prompt_len = roll["prompt_len"]
    gen_len = new_ids.size(1)
    action_slice = slice(prompt_len-1, prompt_len+gen_len-1)  # summary tokens indices - 1

    hidden = model.hidden(roll["full_ids"], roll["full_mask"])  # (batch, prompt_len+gen_len, hidden_size)
    logits = model.policy.lm_head(hidden[:, action_slice])  # (batch, gen_len, vocab)
    new_logp = -torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), new_ids.reshape(-1), reduction="none"
    ).view(new_ids.size(0), gen_len)  # (batch, gen_len)

    values = model.values(hidden)[:, action_slice]  # (batch, gen_len)

    log_ratio = (new_logp - roll["old_logp"]).masked_fill(~keep, 0.0)  # (batch, gen_len)
    ratio = torch.exp(log_ratio)  # (batch, gen_len)

    keep_f = keep.to(ratio.dtype)  # (batch, gen_len)
    denom = keep_f.sum().clamp(min=1)  # scalar containing the count of valid tokens at least have to be 1

    unclipped = ratio * advantages  # (batch, gen_len)
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages  # (batch, gen_len)
    pi_loss = -((torch.min(unclipped, clipped) * keep_f).sum() / denom)

    v_loss = (((values - returns) ** 2) * keep_f).sum() / denom

    log_probs = torch.log_softmax(logits, dim=-1)  # (batch, gen_len, vocab)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)  # (batch, gen_len)
    h_loss = (entropy * keep_f).sum() / denom

    loss = pi_loss + c_v * v_loss - c_h * h_loss

    with torch.no_grad():
        outside = ((ratio < 1 - clip_eps) | (ratio > 1 + clip_eps)).to(keep_f.dtype)
        clip_frac = (outside * keep_f).sum() / denom
        approx_kl = ((roll["old_logp"] - new_logp) * keep_f).sum() / denom

    stats = {
        "pi_loss": pi_loss.item(), 
        "v_loss": v_loss.item(), 
        "h_loss": h_loss.item(),
        "clip_frac": clip_frac.item(), 
        "approx_kl": approx_kl.item(), 
        "loss": loss.item()
    }

    return loss, stats


def train_ppo(tok, batch_size, model_name_or_path, rwd_model_path, max_new_tokens, do_sample, temperature, top_k, top_p, lr, grad_clip, beta, clip_eps, c_v, c_h, n_iters, ppo_epochs, gamma, chkpnt_dir, fig_dir, device):
    """run ppo for n_iters rollouts, each one updated for ppo_epochs epochs,
    logging the reward, kl, length, clip fraction and the three loss terms

    Args:
        tok (AutoTokenizer): tokenizer used for padding and saving
        batch_size (int): number of prompts sampled per iteration
        model_name_or_path (str): path to checkpoints/sft/, the starting point of the policy and the reference
        rwd_model_path (str): path to checkpoints/reward/
        max_new_tokens (int): maximum summary length in tokens
        do_sample (bool): sample the rollouts instead of greedy decoding
        temperature (float): sampling temperature
        top_k (int): top-k sampling parameter, 0 disables it
        top_p (float): top-p sampling parameter, 1.0 disables it
        lr (float): AdamW learning rate
        grad_clip (float): gradient norm clipping value
        beta (float): beta_KL coefficient
        clip_eps (float): epsilon of the probability ratio clip
        c_v (float): value loss coefficient
        c_h (float): entropy coefficient
        n_iters (int): number of ppo iterations
        ppo_epochs (int): update epochs over the same rollout
        gamma (float): discount factor
        chkpnt_dir (str): directory to save the final policy into, without the value head
        fig_dir (str): directory to save the curves into
        device (str): device to train on

    Returns:
        dict: final mean reward, mean kl, mean length, clip fraction, the three losses
            and elapsed time
    """
    train_ds, _ = load_sft_splits(tok)

    model = PolicyWithValue(model_name_or_path).to(device)
    ref_model = load_frozen(model_name_or_path, device)
    rwd_model = RewardModel.load(rwd_model_path, device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    mean_rs, mean_kls, mean_lens, clip_fracs = [], [], [], []
    pi_losses, v_losses, h_losses = [], [], []
    prog_bar = tqdm(range(n_iters), desc="Training PPO")

    start_time = time.time()
    for it in prog_bar:
        batch = random.sample(train_ds, batch_size)
        prompts = collate_prompts(batch, pad_id=tok.pad_token_id)

        roll = rollout(model, ref_model, rwd_model, prompts, tok, max_new_tokens, do_sample, temperature, top_k, top_p, device)
        rewards, kl = per_token_reward(roll["old_logp"], roll["ref_logp"], roll["reward_scores"], roll["keep"], beta)
        returns, advantages = returns_and_advantages(rewards, roll["values"], roll["keep"], gamma)

        model.train()
        for k in range(ppo_epochs):
            opt.zero_grad(set_to_none=True)
            loss, stats = ppo_loss(model, roll, returns, advantages, clip_eps, c_v, c_h)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

        mean_r = roll["reward_scores"].mean().item()
        mean_kl = (kl * roll["keep"]).sum(dim=1).mean().item()
        mean_len = roll["keep"].sum(dim=1).float().mean().item()
        clip_frac = stats["clip_frac"]

        mean_rs.append(mean_r)
        mean_kls.append(mean_kl)
        mean_lens.append(mean_len)
        clip_fracs.append(clip_frac)
        pi_losses.append(stats["pi_loss"])
        v_losses.append(stats["v_loss"])
        h_losses.append(stats["h_loss"])

        prog_bar.set_postfix({"r": f"{mean_r:.4f}", "kl": f"{mean_kl:.4f}", "len": f"{mean_len:.1f}", "clip": f"{clip_frac:.4f}"})
        if (it + 1) % 20 == 0:
            print(f"iter {it + 1}/{n_iters} - mean rϕ: {mean_r:.4f} - KL: {mean_kl:.4f} - "
                  f"len: {mean_len:.1f} - clip: {clip_frac:.4f} - "
                  f"losses pi/v/h: {stats['pi_loss']:.4f}/{stats['v_loss']:.4f}/{stats['h_loss']:.4f}")
    end_time = time.time()
    elapsed = str(timedelta(seconds=int(end_time - start_time)))

    Path(chkpnt_dir).mkdir(parents=True, exist_ok=True)
    model.policy.save_pretrained(chkpnt_dir)
    tok.save_pretrained(chkpnt_dir)

    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    _, ((a1, a2), (a3, a4)) = plt.subplots(2, 2)
    a1.plot(mean_rs)
    a1.set_xlabel("iterations")
    a1.set_title("Mean rϕ Score")
    a2.plot(mean_kls)
    a2.set_xlabel("iterations")
    a2.set_title("Mean KL to Reference")
    a3.plot(mean_lens)
    a3.set_xlabel("iterations")
    a3.set_title("Mean Summary Length")
    a4.plot(clip_fracs)
    a4.set_xlabel("iterations")
    a4.set_title("Clip Fraction")
    plt.suptitle("PPO Training")
    plt.savefig(f"{fig_dir}/ppo_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "mean_r": mean_rs[-1] if mean_rs else 0.0,
        "mean_kl": mean_kls[-1] if mean_kls else 0.0,
        "mean_len": mean_lens[-1] if mean_lens else 0.0,
        "clip_frac": clip_fracs[-1] if clip_fracs else 0.0,
        "losses": (pi_losses[-1], v_losses[-1], h_losses[-1]) if pi_losses else (0.0, 0.0, 0.0),
        "elapsed_time": elapsed,
    }