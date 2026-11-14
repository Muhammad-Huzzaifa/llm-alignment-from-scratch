from src.config import DEVICE, DPO_DIR, PPO_DIR, RESULTS_DIR, RWD_DIR, SFT_DIR
from src.evaluation import run_evaluation
from src.models import load_tokenizer


def main():
    """compare sft, ppo and dpo on the held-out posts,
    writing final_comparison.csv, eval_summaries.jsonl and samples.md into results/

    Returns:
        dict: mean reward, win rate, kl, length and rouge-l for every policy
    """
    tok = load_tokenizer()
    return run_evaluation(
        tok, SFT_DIR, PPO_DIR, DPO_DIR, RWD_DIR, DEVICE, results_dir=RESULTS_DIR,
    )


if __name__ == "__main__":
    main()
