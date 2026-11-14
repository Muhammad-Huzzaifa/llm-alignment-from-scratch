from huggingface_hub import HfApi
from src.config import CHKPT_DIR, HF_TOKEN

REPO_NAME = "llm-alignment-from-scratch"


def push_to_hub(repo_name=REPO_NAME):
    """upload every stage checkpoint inside checkpoints/ to the root of a huggingface model repo,
    so the repo holds sft/, reward/, ppo/ and dpo/ directly instead of a checkpoints/ folder

    Args:
        repo_name (str): name of the huggingface repo under the token's account

    Returns:
        str: the full repo id that was pushed to
    """
    entries = sorted(CHKPT_DIR.iterdir()) if CHKPT_DIR.exists() else []
    if not entries:
        raise FileNotFoundError(f"no checkpoints found in {CHKPT_DIR}, train a stage first")

    api = HfApi(token=HF_TOKEN)
    repo_id = f"{api.whoami()['name']}/{repo_name}"
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    print(f"pushing {CHKPT_DIR} -> https://huggingface.co/{repo_id}")

    for entry in entries:
        if entry.is_dir():  # sft/, reward/, ppo/ and dpo/ end up at the repo root
            api.upload_folder(
                repo_id=repo_id, folder_path=entry, path_in_repo=entry.name,
                commit_message=f"upload {entry.name} checkpoint", repo_type="model",
            )
        else:  # stray files inside checkpoints/ keep their name at the repo root
            api.upload_file(
                repo_id=repo_id, path_or_fileobj=entry, path_in_repo=entry.name,
                commit_message=f"upload {entry.name}", repo_type="model",
            )
        print(f"uploaded {entry.name}")

    return repo_id


def main():
    """push all trained checkpoints to hugging face"""
    repo_id = push_to_hub()
    print(f"done: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
