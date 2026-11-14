import gc
import threading
import time

import torch

from src.config import DEVICE, DPO_DIR, MAX_PROMPT_TOKENS, MAX_SUMMARY_TOKENS, PPO_DIR, RWD_DIR, SFT_DIR
from src.data import clean_post
from src.evaluation import policy_summaries, score_summaries
from src.models import load_causal_lm, load_tokenizer
from src.reward import RewardModel

POLICY_DIRS = {"sft": SFT_DIR, "ppo": PPO_DIR, "dpo": DPO_DIR}
METHODS = list(POLICY_DIRS)

_lock = threading.Lock()  # only one policy should sit in memory at a time
_tok = None
_rwd = None


def _get_tokenizer():
    """load the tokenizer once and reuse it for every request

    Returns:
        AutoTokenizer: the shared tokenizer
    """
    global _tok
    if _tok is None:
        _tok = load_tokenizer()
    return _tok


def _get_reward_model():
    """load the reward model once and keep it in memory, every summary needs a score

    Returns:
        RewardModel: the shared frozen reward model
    """
    global _rwd
    if _rwd is None:
        _rwd = RewardModel.load(RWD_DIR, DEVICE)
    return _rwd


def _prompt_ids(post, tok):
    """format the raw post like the training prompts and tokenize it

    Args:
        post (str): text typed by the user, optionally already in SUBREDDIT/TITLE/POST form
        tok (AutoTokenizer): tokenizer used for encoding

    Returns:
        list: prompt token ids ending with the "TL;DR:" marker
    """
    ids = tok(clean_post(post), add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError("the post is empty")
    if len(ids) > MAX_PROMPT_TOKENS:
        raise ValueError(f"the post is too long: {len(ids)} tokens (max {MAX_PROMPT_TOKENS})")
    return ids


def summarize(post, methods):
    """generate a summary with each requested policy, one policy at a time,
    and score every summary with the same reward model

    Args:
        post (str): raw reddit post typed by the user
        methods (list): ordered subset of "sft", "ppo", "dpo"

    Returns:
        list: one dict per method with "method", "summary", "reward", "length" and "seconds"
    """
    methods = [m.strip().lower() for m in dict.fromkeys(methods) if m and m.strip()]
    unknown = [m for m in methods if m not in POLICY_DIRS]
    if unknown:
        raise ValueError(f"unknown method(s): {', '.join(unknown)}")
    if not methods:
        raise ValueError("pick at least one of SFT, PPO or DPO")

    with _lock:
        tok = _get_tokenizer()
        prompt = [{"prompt_ids": _prompt_ids(post, tok)}]
        rwd = _get_reward_model()
        results = []

        for name in methods:
            start = time.time()
            model = load_causal_lm(POLICY_DIRS[name], DEVICE)
            summary = policy_summaries(model, tok, prompt, 1, MAX_SUMMARY_TOKENS, DEVICE)[0]
            del model  # free the policy before scoring and before loading the next one
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            reward = score_summaries(rwd, tok, prompt, [summary], 1, DEVICE)[0]
            results.append({
                "method": name,
                "summary": summary["text"].strip(),
                "reward": round(float(reward), 4),
                "length": len(summary["ids"]),
                "seconds": round(time.time() - start, 1),
            })

        return results
