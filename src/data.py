import random
from datasets import load_dataset
from src.config import *

TLDR = "TL;DR:"


def clean_post(raw_prompt):
    """keep SUBREDDIT/TITLE/POST, drop anything from 'TL;DR:' on, and end with one fixed marker the model learns to continue
    
    Args:
        raw_prompt (str): the raw prompt text from the dataset

    Returns:
        str: cleaned prompt text with a fixed marker at the end
    """
    post = raw_prompt.split(TLDR)[0].strip()
    return post + "\n" + TLDR


def clean_summary(raw):
    """strip a leading 'TL;DR:', cut stray text after <|endoftext|>, and start with exactly one space (it follows 'TL;DR:')
    
    Args:
        raw (str): the raw summary text from the dataset

    Returns:
        str: cleaned summary text
    """
    text = raw.split("<|endoftext|>")[0].strip()
    if text.startswith(TLDR):
        text = text[len(TLDR):].strip()
    return " " + text


def _sample_valid(rows, n, tok, make, seed):
    """shuffle rows with a fixed seed and keep the first n that pass the length filter, each kept item is a dict of token-id lists
    
    Args:
        rows (list): list of dataset rows
        n (int): number of valid items to keep
        tok (AutoTokenizer): tokenizer to convert text to token ids
        make (function): function to create a dict of token-id lists from a row
        seed (int): random seed for shuffling

    Returns:
        list: list of valid items (dicts of token-id lists)
    """
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    kept = []
    for start in range(0, len(order), 2000):
        for i in order[start:start + 2000]:
            item = make(rows[i], tok)
            if item is not None:
                kept.append(item)
                if len(kept) == n:
                    return kept
    return kept


def _make_sft(row, tok):
    """create a dict of token-id lists from a row, or return None if the prompt or summary is too long or empty

    Args:
        row (dict): a dataset row with "prompt" and "label" keys
        tok (AutoTokenizer): tokenizer to convert text to token ids

    Returns:
        dict or None: a dict with token-id lists for "prompt_ids" and "summary_ids", or None if the prompt or summary is too long or empty
    """
    prompt, summary = clean_post(row["prompt"]), clean_summary(row["label"])
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    s = tok(summary, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    if len(summary.strip()) == 0 or len(p) > MAX_PROMPT_TOKENS or len(s) > MAX_SUMMARY_TOKENS:
        return None
    return {"prompt_ids": p, "summary_ids": s, "prompt": prompt, "summary": summary}


def _make_pref(row, tok):
    """create a dict of token-id lists from a row, or return None if the prompt or summaries are too long or empty

    Args:
        row (dict): a dataset row with "prompt", "chosen", and "rejected" keys
        tok (AutoTokenizer): tokenizer to convert text to token ids

    Returns:
        dict or None: a dict with token-id lists for "prompt_ids", "chosen_ids", and "rejected_ids", or None if any of the texts are too long or empty
    """
    prompt = clean_post(row["prompt"])
    ch, rj = clean_summary(row["chosen"]), clean_summary(row["rejected"])
    if ch == rj or not ch.strip() or not rj.strip():
        return None
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    c = tok(ch, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    r = tok(rj, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    if len(p) > MAX_PROMPT_TOKENS or max(len(c), len(r)) > MAX_SUMMARY_TOKENS:
        return None
    return {"prompt_ids": p, "chosen_ids": c, "rejected_ids": r, "prompt": prompt, "chosen": ch, "rejected": rj}


def load_sft_splits(tok):
    """load the SFT dataset and return the train and validation splits, each as a list of dicts of token-id lists
    
    Args:
        tok (AutoTokenizer): tokenizer to convert text to token ids

    Returns:
        tuple: (train, val) where train and val are lists of dicts of token-id lists
    """
    ds = load_dataset(SFT_DATA, cache_dir=str(DATA_DIR), token=HF_TOKEN)
    train = _sample_valid(ds["train"], N_SFT_TRAIN, tok, _make_sft, SEED)
    val = _sample_valid(ds["valid"], N_SFT_VAL, tok, _make_sft, SEED)
    return train, val

def load_preference_splits(tok):
    """load the preference dataset and return the train and validation splits, each as a list of dicts of token-id lists

    Args:
        tok (AutoTokenizer): tokenizer to convert text to token ids

    Returns:
        tuple: (train, val) where train and val are lists of dicts of token-id lists
    """
    ds = load_dataset(PREF_DATA, cache_dir=str(DATA_DIR), token=HF_TOKEN)
    train = _sample_valid(ds["train"], N_PREF_TRAIN, tok, _make_pref, SEED)
    val = _sample_valid(ds["valid1"], N_PREF_VAL, tok, _make_pref, SEED)
    return train, val

def load_eval_posts(tok):
    """held-out posts (with their human reference summary) used to compare SFT, PPO and DPO at the end
    
    Args:
        tok (AutoTokenizer): tokenizer to convert text to token ids

    Returns:
        list: list of dicts of token-id lists for evaluation posts
    """
    ds = load_dataset(SFT_DATA, cache_dir=str(DATA_DIR), token=HF_TOKEN)
    return _sample_valid(ds["test"], N_EVAL_POSTS, tok, _make_sft, SEED)
