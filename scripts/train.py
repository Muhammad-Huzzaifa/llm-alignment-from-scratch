import argparse
import csv
from src.config import *
from src.models import load_tokenizer
from src.data import load_sft_splits, load_preference_splits, load_eval_posts
from src.sft import train_sft
from src.reward import train_reward
from src.ppo import train_ppo
from src.dpo import train_dpo

STAGES = ["sft", "reward", "ppo", "dpo"]
STAGE_NAMES = {"sft": "SFT", "reward": "Reward model", "ppo": "PPO", "dpo": "DPO"}

# ppo rollout sampling and discount, fixed for this assignment
PPO_DO_SAMPLE, PPO_TEMPERATURE, PPO_TOP_K, PPO_TOP_P = True, 1.0, 0, 1.0
PPO_GAMMA = 1.0


def write_data_stats(tok):
    """size and token-length statistics of the three splits, written to results/data_stats.csv

    Args:
        tok (AutoTokenizer): tokenizer used to load and measure the splits
    """
    sft_train, sft_val = load_sft_splits(tok)
    pref_train, pref_val = load_preference_splits(tok)
    eval_posts = load_eval_posts(tok)

    prompt_lens = [len(x["prompt_ids"]) for x in sft_train]
    summary_lens = [len(x["summary_ids"]) for x in sft_train]

    rows = [
        ["SFT train / validation pairs", f"{len(sft_train)} / {len(sft_val)}"],
        ["Preference train / validation pairs", f"{len(pref_train)} / {len(pref_val)}"],
        ["Evaluation posts", len(eval_posts)],
        ["Mean / max prompt length, SFT train (tokens)", f"{sum(prompt_lens) / len(prompt_lens):.2f} / {max(prompt_lens)}"],
        ["Mean / max summary length, SFT train (tokens)", f"{sum(summary_lens) / len(summary_lens):.2f} / {max(summary_lens)}"],
    ]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "data_stats.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Item", "Value"])
        w.writerows(rows)


def write_training_stats(stage, stats):
    """append one stage row to results/training_stats.csv, replacing the stage's old row if it is re-run

    Args:
        stage (str): "SFT", "Reward model", "PPO" or "DPO"
        stats (dict): whatever train_sft, train_reward, train_ppo or train_dpo returned
    """
    if stage == "SFT":
        train_loss, metric = stats["train_loss"], f"val. loss: {stats['val_loss']:.4f}"
    elif stage == "Reward model":
        train_loss, metric = stats["train_loss"], f"pairwise acc.: {stats['pairwise_acc']:.4f}"
    elif stage == "PPO":
        train_loss, metric = stats["losses"][0], f"final mean rϕ: {stats['mean_r']:.4f}"  # pi_loss is the ppo training loss
    else:
        train_loss, metric = stats["train_loss"], f"implicit reward acc.: {stats['imp_rwd_acc']:.4f}"

    row = [stage, f"{train_loss:.4f}", metric, f"{stats['elapsed_time']} / {MACHINE_NAME}"]
    order = {name: i for i, name in enumerate(STAGE_NAMES.values())}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "training_stats.csv"
    rows = []
    if path.exists():
        with open(path, newline="") as f:
            rows = [r for r in csv.reader(f)][1:]
    rows = [r for r in rows if r and r[0] != stage] + [row]
    rows.sort(key=lambda r: order.get(r[0], len(order)))

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Stage", "Final train loss", "Validation metric", "Time / GPU"])
        w.writerows(rows)


def write_human_eval_template():
    """write an empty results/human_eval.csv for the two raters, if it does not exist yet"""
    path = RESULTS_DIR / "human_eval.csv"
    if path.exists():
        return
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Member", "SFT Picked", "PPO Picked", "DPO Picked"])
        w.writerow(["Member 1", "", "", ""])
        w.writerow(["Member 2", "", "", ""])
        w.writerow(["Agreement between members", "", "", "___/30"])


def train(model_name):
    """train one stage with the config hyperparameters and paths, then record its metrics

    Args:
        model_name (str): one of "sft", "reward", "ppo" or "dpo"

    Returns:
        dict: whatever the stage's train_* function returned
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    tok = load_tokenizer()

    if not (RESULTS_DIR / "data_stats.csv").exists():
        write_data_stats(tok)
    write_human_eval_template()

    if model_name == "sft":
        stats = train_sft(
            tok=tok, batch_size=SFT_BS, model_name_or_path=MODEL_NAME, lr=SFT_LR,
            grad_clip=GRAD_CLIP, chkpnt_dir=SFT_DIR, fig_dir=FIG_DIR, device=DEVICE,
        )
    elif model_name == "reward":
        stats = train_reward(
            tok=tok, batch_size=RWD_BS, model_name_or_path=SFT_DIR, lr=RWD_LR,
            grad_clip=GRAD_CLIP, chkpnt_dir=RWD_DIR, fig_dir=FIG_DIR, device=DEVICE,
        )
    elif model_name == "ppo":
        stats = train_ppo(
            tok=tok, batch_size=PPO_PROMPTS, model_name_or_path=SFT_DIR, rwd_model_path=RWD_DIR,
            max_new_tokens=MAX_SUMMARY_TOKENS, do_sample=PPO_DO_SAMPLE,
            temperature=PPO_TEMPERATURE, top_k=PPO_TOP_K, top_p=PPO_TOP_P, lr=PPO_LR,
            grad_clip=GRAD_CLIP, beta=PPO_BETA_KL, clip_eps=PPO_CLIP, c_v=PPO_C_V, c_h=PPO_C_H,
            n_iters=PPO_ITERS, ppo_epochs=PPO_EPOCHS, gamma=PPO_GAMMA,
            chkpnt_dir=PPO_DIR, fig_dir=FIG_DIR, device=DEVICE,
        )
    elif model_name == "dpo":
        stats = train_dpo(
            tok=tok, batch_size=DPO_BS, model_name_or_path=SFT_DIR, lr=DPO_LR,
            grad_clip=GRAD_CLIP, beta=DPO_BETA, chkpnt_dir=DPO_DIR, fig_dir=FIG_DIR, device=DEVICE,
        )
    else:
        raise ValueError(f"unknown model '{model_name}', choose from {STAGES}")

    write_training_stats(STAGE_NAMES[model_name], stats)
    print(f"{STAGE_NAMES[model_name]} done: {stats}")

    return stats


def parse_args(argv=None):
    """parse the --model stage argument shared by local_train.py and modal_train.py

    Args:
        argv (list): argument list to parse, sys.argv[1:] when None

    Returns:
        argparse.Namespace: parsed arguments with "model" in STAGES
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=STAGES, help="stage to train")
    return parser.parse_args(argv)


def main(argv=None):
    """train the stage given on the command line

    Args:
        argv (list): argument list to parse, sys.argv[1:] when None
    """
    train(parse_args(argv).model)


if __name__ == "__main__":
    main()
