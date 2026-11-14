import os
from pathlib import Path
from dotenv import load_dotenv
import torch

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"  # base model for every stage

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
CHKPT_DIR = ROOT / "checkpoints"
SFT_DIR = CHKPT_DIR / "sft"
RWD_DIR = CHKPT_DIR / "reward"
PPO_DIR = CHKPT_DIR / "ppo"
DPO_DIR = CHKPT_DIR / "dpo"

HF_TOKEN = os.getenv("HF_TOKEN")
SFT_DATA = "CarperAI/openai_summarize_tldr"
PREF_DATA = "CarperAI/openai_summarize_comparisons"

MAX_PROMPT_TOKENS = 448  # post + "TL;DR:" longer than this is dropped
MAX_SUMMARY_TOKENS = 48  # summaries longer than this are dropped

SEED = 42
N_SFT_TRAIN, N_SFT_VAL = 20_000, 500
N_PREF_TRAIN, N_PREF_VAL = 20_000, 1_000
N_EVAL_POSTS = 300

SFT_LR = 5e-5
SFT_BS = 16
RWD_LR = 1e-5
RWD_BS = 8
PPO_LR = 1e-5
PPO_CLIP = 0.2
PPO_BETA_KL = 0.05
PPO_C_V = 0.1
PPO_C_H = 0.01
PPO_EPOCHS = 4
PPO_ITERS = 200
PPO_PROMPTS = 16
DPO_LR = 2e-6
DPO_BS = 8
DPO_BETA = 0.1
GRAD_CLIP = 1.0

MODEL_DTYPE = torch.float32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MACHINE_NAME = torch.cuda.get_device_name(DEVICE) if DEVICE.type == "cuda" else "CPU"