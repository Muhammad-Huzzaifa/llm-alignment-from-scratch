import torch


def _pad(seqs, pad_id, side):
    """pad a list of sequences to the same length with the given pad_id, either on the left or right side

    Args:
        seqs (list of list of int): list of sequences to pad
        pad_id (int): the token id to use for padding
        side (str): "left" or "right", indicating which side to pad

    Returns:
        tuple: (ids, mask) where ids is a tensor of shape (batch_size, max_len) with padded sequences, and mask is a tensor of the same shape with 1s for real tokens and 0s for padding
    """
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        t = torch.tensor(s, dtype=torch.long)
        if side == "right":
            ids[i, :len(s)], mask[i, :len(s)] = t, 1
        else:
            # left padding
            ids[i, L - len(s):], mask[i, L - len(s):] = t, 1
    return ids, mask


def collate_sft(batch, pad_id):
    """right-padded prompt+summary
    prompt_lens[i] = number of prompt
    tokens in row i, so summary tokens are positions prompt_lens[i]

    Args:
        batch (list): list of dicts with "prompt_ids" and "summary_ids"
        pad_id (int): the token id to use for padding

    Returns:
        dict: a dict with keys "input_ids", "attention_mask", and "prompt_lens", 
            where "input_ids" is a tensor of shape (batch_size, max_len) with padded sequences, 
            "attention_mask" is a tensor of the same shape with 1s for real tokens and 0s for padding, 
            and "prompt_lens" is a tensor of shape (batch_size,) with the lengths of the prompts
    """
    seqs = [b["prompt_ids"] + b["summary_ids"] for b in batch]
    ids, mask = _pad(seqs, pad_id, "right")
    prompt_lens = torch.tensor([len(b["prompt_ids"]) for b in batch])
    return {"input_ids": ids, "attention_mask": mask, "prompt_lens": prompt_lens}


def collate_pref(batch, pad_id):
    """two right-padded batches (prompt+chosen, prompt+rejected) that share the same prompt_lens
    
    Args:
        batch (list): list of dicts with "prompt_ids" and "summary_ids"
        pad_id (int): the token id to use for padding
    
    Returns:
        dict: a dict with keys "chosen_ids", "chosen_mask", "rejected_ids", "rejected_mask", and "prompt_lens", 
            where "chosen_ids" and "rejected_ids" are tensors of shape (batch_size, max_len) with padded sequences, 
            "chosen_mask" and "rejected_mask" are tensors of the same shape with 1s for real tokens and 0s for padding, 
            and "prompt_lens" is a tensor of shape (batch_size,) with the lengths of the prompts
    """
    ch, ch_mask = _pad([b["prompt_ids"] + b["chosen_ids"] for b in batch], pad_id, "right")
    rj, rj_mask = _pad([b["prompt_ids"] + b["rejected_ids"] for b in batch], pad_id, "right")
    prompt_lens = torch.tensor([len(b["prompt_ids"]) for b in batch])
    return {
        "chosen_ids": ch, "chosen_mask": ch_mask,
        "rejected_ids": rj, "rejected_mask": rj_mask,
        "prompt_lens": prompt_lens
    }


def collate_prompts(batch, pad_id):
    """left-padded prompts only, 
    for model.generate() in PPO rollouts and evaluation,
    left padding keeps every prompt's last token at the same position, 
    which is where generation starts
    
    Args:
        batch (list): list of dicts with "prompt_ids"
        pad_id (int): the token id to use for padding

    Returns:
        dict: a dict with keys "input_ids" and "attention_mask",
            where "input_ids" is a tensor of shape (batch_size, max_len) with padded sequences, 
            and "attention_mask" is a tensor of the same shape with 1
    """
    ids, mask = _pad([b["prompt_ids"] for b in batch], pad_id, "left")
    return {"input_ids": ids, "attention_mask": mask}
