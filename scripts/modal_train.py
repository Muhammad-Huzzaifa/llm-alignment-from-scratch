import argparse
import subprocess
import sys
import traceback
from pathlib import Path
import modal
from src.config import ROOT, CHKPT_DIR, RESULTS_DIR

REMOTE_ROOT = "/root/llm-alignment-from-scratch"
VOL_DIR = "/root/vol"
VOLUME_NAME = "llm-alignment-from-scratch"
STAGES = ["sft", "reward", "ppo", "dpo"]
GPU = "A100-40GB"
TIMEOUT = 24 * 60 * 60

app = modal.App("llm-alignment-from-scratch")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
secret = modal.Secret.from_dotenv(ROOT)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(ROOT / "requirements.txt"))
    .env({"PYTHONPATH": f"{REMOTE_ROOT}:{REMOTE_ROOT}/scripts:/root"})
    .workdir(REMOTE_ROOT)
    .add_local_dir(str(ROOT / "src"), f"{REMOTE_ROOT}/src")
    .add_local_dir(str(ROOT / "scripts"), f"{REMOTE_ROOT}/scripts")
)


@app.function(image=image, volumes={VOL_DIR: volume}, secrets=[secret], gpu=GPU, timeout=TIMEOUT)
def run_remote(model: str = "", evaluate: bool = False):
    """train one stage or run the evaluation inside the modal container,
    with the volume mounted as the project's checkpoints and results

    Args:
        model (str): stage to train, one of "sft", "reward", "ppo" or "dpo"
        evaluate (bool): run scripts/evaluate.py instead of a training stage

    Returns:
        dict: {"success": bool, "error": str or None, "files": volume paths written by the stage}
    """
    import torch

    project, vol = Path(REMOTE_ROOT), Path(VOL_DIR)
    for name in ("checkpoints", "results"):  # config.py resolves these next to src/
        (vol / name).mkdir(parents=True, exist_ok=True)
        link = project / name
        if link.is_symlink():
            link.unlink()
        link.symlink_to(vol / name)

    if evaluate:
        cmd = [sys.executable, "scripts/evaluate.py"]
    else:
        cmd = [sys.executable, "scripts/train.py", "--model", model]

    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    print("running:", " ".join(cmd))

    error_text = None
    try:
        subprocess.run(cmd, cwd=project, check=True)
    except Exception:
        error_text = traceback.format_exc()  # keep going so the local side can still pull partial outputs
        print(error_text)
    finally:
        volume.commit()  # persist whatever the stage wrote, even when it failed

    wanted = ["results"] if evaluate else ["results", f"checkpoints/{model}"]
    files = sorted(
        path.relative_to(vol).as_posix()
        for folder in wanted
        for path in (vol / folder).rglob("*")
        if path.is_file()
    )
    return {"success": error_text is None, "error": error_text, "files": files}


def download(files):
    """copy the files written by run_remote from the modal volume into the local project

    Args:
        files (list): volume paths such as "results/final_comparison.csv" or "checkpoints/sft/model.safetensors"
    """
    for name in files:
        target = ROOT / name  # the volume mirrors the project layout, so results/ and checkpoints/ line up
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as f:
            for chunk in volume.read_file(name):
                f.write(chunk)
        print(f"downloaded {name} -> {target}")


def run(model: str, evaluate: bool):
    """train one stage or run the evaluation on modal, then download its outputs

    Args:
        model (str): stage to train, one of "sft", "reward", "ppo" or "dpo"
        evaluate (bool): run the evaluation instead of a training stage
    """
    if evaluate and model:
        raise ValueError("pass either --model or --evaluate, not both")
    if not evaluate and model not in STAGES:
        raise ValueError(f"pass --model {{{','.join(STAGES)}}} or --evaluate")

    print(f"launching modal {GPU} for {'evaluation' if evaluate else model}...")
    outcome = run_remote.remote(model, evaluate)
    download(outcome["files"])  # partial outputs are pulled even when the stage failed
    if not outcome["success"]:  # the volume is not cleared, unlike help.py, so training_stats.csv keeps every stage
        raise RuntimeError(
            f"remote run failed (model={model or 'evaluate'}). partial outputs, if any, were pulled into "
            f"{RESULTS_DIR.resolve()} and {CHKPT_DIR.resolve()}.\n\n{outcome['error']}"
        )
    print(f"done, {len(outcome['files'])} files pulled into {ROOT.resolve()}")


@app.local_entrypoint()
def main(model: str = "", evaluate: bool = False):
    """run one training stage or the evaluation on modal, then download the outputs

    Args:
        model (str): stage to train, one of "sft", "reward", "ppo" or "dpo"
        evaluate (bool): run the evaluation instead of a training stage
    """
    run(model, evaluate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", choices=STAGES, help="stage to train on modal")
    group.add_argument("--evaluate", action="store_true", help="run the evaluation on modal")
    args = parser.parse_args()

    with modal.enable_output():
        with app.run():
            run(args.model or "", args.evaluate)
