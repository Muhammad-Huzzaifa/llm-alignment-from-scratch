import csv
import json
from functools import partial
from pathlib import Path
from src.batching import collate_prompts, collate_sft
from src.common import generate_summaries, sequence_logprobs
from src.config import MAX_SUMMARY_TOKENS, RESULTS_DIR
from src.data import load_eval_posts
from src.models import load_causal_lm, load_frozen
from src.reward import RewardModel
import torch
from tqdm.auto import tqdm
from rouge_score import rouge_scorer


def _summary_batches(tok, eval_posts, summaries, batch_size):
    """right-padded prompt+summary batches, so one dataloader can feed both the reward model and the kl

    Args:
        tok (AutoTokenizer): tokenizer used for padding
        eval_posts (list): load_eval_posts output, each dict has "prompt_ids"
        summaries (list): policy_summaries output, each dict has "ids"
        batch_size (int): number of posts per batch

    Returns:
        DataLoader: batches with "input_ids", "attention_mask" and "prompt_lens"
    """
    items = [{"prompt_ids": p["prompt_ids"], "summary_ids": s["ids"]} for p, s in zip(eval_posts, summaries)]
    return torch.utils.data.DataLoader(items, batch_size=batch_size, shuffle=False, collate_fn=partial(collate_sft, pad_id=tok.pad_token_id))


def policy_summaries(model, tok, eval_posts, batch_size, max_new_tokens, device):
    """greedily generate one summary per post with a single policy,
    cut each one at its first eos so nothing after it is counted

    Args:
        model (AutoModelForCausalLM): the policy to generate with (sft, ppo or dpo)
        tok (AutoTokenizer): tokenizer used for encoding and decoding
        eval_posts (list): load_eval_posts output, each dict has "prompt_ids"
        batch_size (int): number of posts per generation batch
        max_new_tokens (int): maximum summary length in tokens
        device (str): device to run generation on

    Returns:
        list: one dict per post with "ids" (up to the first eos) and the decoded "text"
    """
    dl = torch.utils.data.DataLoader(eval_posts, batch_size=batch_size, shuffle=False, collate_fn=partial(collate_prompts, pad_id=tok.pad_token_id))

    model.eval()
    out = []
    with torch.no_grad():
        for batch in tqdm(dl, desc="Generating", leave=False):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            new_ids = generate_summaries(model, tok, ids, mask, max_new_tokens, do_sample=False)  # (batch, gen_len)

            for row in new_ids:
                is_eos = (row == tok.eos_token_id).long()  # [a, b, eos, c] -> [0, 0, 1, 0]
                keep = (is_eos.cumsum(dim=0) - is_eos) == 0  # [0, 0, 0, 1] -> 1 up to and including the first eos
                kept = row[keep].tolist()
                out.append({"ids": kept, "text": tok.decode(kept, skip_special_tokens=True)})

    return out


def score_summaries(rwd_model, tok, eval_posts, summaries, batch_size, device):
    """r_phi score of every generated summary with the frozen reward model

    Args:
        rwd_model (RewardModel): frozen reward model r_phi
        tok (AutoTokenizer): tokenizer used for padding
        eval_posts (list): load_eval_posts output, each dict has "prompt_ids"
        summaries (list): policy_summaries output, each dict has "ids"
        batch_size (int): number of posts per batch
        device (str): device to run scoring on

    Returns:
        list of float: one reward per post
    """
    rwd_model.eval()
    scores = []
    with torch.no_grad():
        for batch in _summary_batches(tok, eval_posts, summaries, batch_size):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            scores += rwd_model(ids, mask).squeeze(-1).tolist()  # (batch,)

    return scores


def kl_to_sft(model, ref_model, tok, eval_posts, summaries, batch_size, device):
    """sum over the summary tokens of log pi_theta - log pi_ref, on the policy's own summary

    Args:
        model (AutoModelForCausalLM): the policy pi_theta (ppo or dpo)
        ref_model (AutoModelForCausalLM): frozen sft reference pi_ref
        tok (AutoTokenizer): tokenizer used for padding
        eval_posts (list): load_eval_posts output, each dict has "prompt_ids"
        summaries (list): the policy's own policy_summaries output
        batch_size (int): number of posts per batch
        device (str): device to run the forward passes on

    Returns:
        list of float: one kl per post
    """
    model.eval()
    ref_model.eval()
    kls = []
    with torch.no_grad():
        for batch in _summary_batches(tok, eval_posts, summaries, batch_size):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            prompt_lens = batch["prompt_lens"].to(device)
            kl = sequence_logprobs(model, ids, mask, prompt_lens) - sequence_logprobs(ref_model, ids, mask, prompt_lens)  # (batch,)
            kls += kl.tolist()

    return kls


def run_evaluation(tok, sft_path, ppo_path, dpo_path, rwd_path, device, results_dir=RESULTS_DIR, batch_size=16, max_new_tokens=MAX_SUMMARY_TOKENS):
    """compare sft, ppo and dpo on the 300 held-out posts,
    writing final_comparison.csv, eval_summaries.jsonl and samples.md

    Args:
        tok (AutoTokenizer): tokenizer used for padding and decoding
        sft_path (str): path to checkpoints/sft/
        ppo_path (str): path to checkpoints/ppo/
        dpo_path (str): path to checkpoints/dpo/
        rwd_path (str): path to checkpoints/reward/
        device (str): device to run evaluation on
        results_dir (str): directory to write the results into
        batch_size (int): number of posts per batch
        max_new_tokens (int): maximum summary length in tokens

    Returns:
        dict: mean reward, win rate, kl, length and rouge-l for every policy
    """
    eval_posts = load_eval_posts(tok)
    ref_model = load_frozen(sft_path, device)  # also the sft policy, generating on a frozen model is fine
    rwd_model = RewardModel.load(rwd_path, device)

    policies = ["SFT", "PPO", "DPO"]
    paths = {"SFT": None, "PPO": ppo_path, "DPO": dpo_path}
    summaries, scores, kls = {}, {}, {}

    for name in policies:
        model = ref_model if paths[name] is None else load_causal_lm(paths[name], device)

        summaries[name] = policy_summaries(model, tok, eval_posts, batch_size, max_new_tokens, device)
        scores[name] = score_summaries(rwd_model, tok, eval_posts, summaries[name], batch_size, device)
        kls[name] = [0.0] * len(eval_posts) if name == "SFT" else kl_to_sft(model, ref_model, tok, eval_posts, summaries[name], batch_size, device)  # sft is its own reference
        print(f"{name} - mean r_phi: {sum(scores[name]) / len(scores[name]):.4f}")

        if model is not ref_model:  # keep only the reference and the reward model in gpu memory
            del model
            torch.cuda.empty_cache()

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    metrics = {}
    for name in policies:
        wins = sum((a > b) + 0.5 * (a == b) for a, b in zip(scores[name], scores["SFT"])) / len(eval_posts)  # a tie counts as half a win
        metrics[name] = {
            "reward": sum(scores[name]) / len(scores[name]),
            "win_rate": wins,
            "kl": sum(kls[name]) / len(kls[name]),
            "length": sum(len(s["ids"]) for s in summaries[name]) / len(summaries[name]),
            "rouge": sum(scorer.score(p["summary"], s["text"])["rougeL"].fmeasure for p, s in zip(eval_posts, summaries[name])) / len(eval_posts),
        }

    Path(results_dir).mkdir(parents=True, exist_ok=True)

    with open(Path(results_dir) / "final_comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Policy", "Mean rϕ", "Win rate vs. SFT", "KL to SFT", "Length", "ROUGE-L"])
        for name in policies:
            m = metrics[name]
            w.writerow([name, f"{m['reward']:.4f}", "-" if name == "SFT" else f"{m['win_rate']:.4f}", "0" if name == "SFT" else f"{m['kl']:.4f}", f"{m['length']:.2f}", f"{m['rouge']:.4f}"])

    with open(Path(results_dir) / "eval_summaries.jsonl", "w") as f:
        for i, post in enumerate(eval_posts):
            row = {"idx": i, "post": post["prompt"], "reference": post["summary"]}
            for name in policies:
                row[name.lower()] = {"summary": summaries[name][i]["text"], "reward": round(scores[name][i], 4), "length": len(summaries[name][i]["ids"])}
            f.write(json.dumps(row) + "\n")

    deltas = [max(scores["PPO"][i], scores["DPO"][i]) - scores["SFT"][i] for i in range(len(eval_posts))]
    order = sorted(range(len(eval_posts)), key=lambda i: deltas[i])
    picked = order[-5:] + order[:5]  # 5 where alignment helps the most and 5 where it hurts the most

    lines = ["# Samples\n"]
    for i in picked:
        lines.append(f"## Post {i}\n")
        lines.append(f"**Post**\n\n{eval_posts[i]['prompt']}\n")
        lines.append(f"**Human Reference**\n\n{eval_posts[i]['summary'].strip()}\n")
        for name in policies:
            lines.append(f"**{name}** (rϕ = {scores[name][i]:.4f})\n\n{summaries[name][i]['text'].strip()}\n")
    with open(Path(results_dir) / "samples.md", "w") as f:
        f.write("\n".join(lines))

    print(f"{'Policy':<8}{'Mean rϕ':>10}{'Win rate':>10}{'KL':>10}{'Length':>8}{'ROUGE-L':>10}")
    for name in policies:
        m = metrics[name]
        win = "-" if name == "SFT" else f"{m['win_rate']:.4f}"
        print(f"{name:<8}{m['reward']:>10.4f}{win:>10}{m['kl']:>10.4f}{m['length']:>8.2f}{m['rouge']:>10.4f}")

    return metrics
