import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.config import HF_TOKEN, MODEL_NAME, MODEL_DTYPE


def load_tokenizer(name=MODEL_NAME):
    """load the tokenizer for the given model name
    if the model has no pad token, set it to the eos token
    
    Args:
        name (str): the name of the model to load the tokenizer for

    Returns:
        AutoTokenizer: the loaded tokenizer
    """
    tok = AutoTokenizer.from_pretrained(name, token=HF_TOKEN)
    if tok.pad_token is None: # SmolLM2 has no pad token
        tok.pad_token = tok.eos_token
    return tok


def load_causal_lm(name_or_path=MODEL_NAME, device=None):
    """plain pretrained causal LM,
    use it for the SFT policy, the frozen reference model, and as the backbone of the reward model and of the PPO value head
    
    Args:
        name_or_path (str): the name or path of the model to load
        device (str): the device to load the model on (default: cuda if available, else cpu)

    Returns:
        AutoModelForCausalLM: the loaded model with backbone at .model and head at .lm_head
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(name_or_path, token=HF_TOKEN, dtype=MODEL_DTYPE)
    return model.to(device)


def load_frozen(name_or_path, device=None):
    """load a frozen pretrained causal LM,
    use it for the frozen reference model and as the backbone of the reward model and of the PPO value head
    
    Args:
        name_or_path (str): the name or path of the model to load
        device (str): the device to load the model on (default: cuda if available, else cpu)

    Returns:
        AutoModelForCausalLM: the loaded model with backbone at .model and head at .lm_head, with all parameters frozen
    """
    model = load_causal_lm(name_or_path, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model
