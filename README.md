# llm-alignment-from-scratch

custom PyTorch implementation of the full LLM alignment pipeline: SFT → reward model → PPO → DPO, built from scratch on SmolLM2 for the task of learning to summarize from human feedback.

All four stages were trained in float32 on a single NVIDIA A100-SXM4-40GB.

## Structure

```
├── app/                             # fastapi demo (one page, no build step)
│   ├── inference.py                 # load one policy, generate, score, free it
│   ├── main.py                      # routes + results/final_comparison.csv metrics
│   ├── static/
│   │   ├── script.js
│   │   └── style.css
│   └── templates/
│       └── index.html
├── checkpoints/                     # sft/, reward/, ppo/, dpo/ (gitignored)
├── data/                            # Hugging Face dataset cache (gitignored)
├── notebooks/
│   └── llm_alignment_from_scratch_repo_execution.ipynb
├── results/                         # metrics, samples and training curves
├── scripts/
│   ├── evaluate.py
│   ├── local_train.py
│   ├── modal_train.py
│   ├── push_to_hub.py
│   └── train.py
├── src/
│   ├── __init__.py
│   ├── batching.py
│   ├── common.py
│   ├── config.py
│   ├── data.py
│   ├── dpo.py
│   ├── evaluation.py
│   ├── models.py
│   ├── ppo.py
│   ├── reward.py
│   └── sft.py
├── requirements.txt
└── README.md
```

## Setup

```
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root with your Hugging Face token:

```
HF_TOKEN=your_huggingface_token
```

All paths and hyperparameters live in `src/config.py`.

## Demo app

```bash
python -m uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000, paste a post, tick SFT / PPO / DPO and generate. Each policy is loaded only for its own generation and then freed, so the app holds one policy plus the reward model at a time. Summaries come back side by side with their reward score, length and latency.

## Data

`CarperAI/openai_summarize_tldr` for SFT and `CarperAI/openai_summarize_comparisons` for the preference pairs, downloaded through `datasets`.

| item | value |
| --- | --- |
| SFT train / validation pairs | 20000 / 500 |
| Preference train / validation pairs | 20000 / 1000 |
| Evaluation posts | 300 |
| Mean / max prompt length, SFT train (tokens) | 317.42 / 448 |
| Mean / max summary length, SFT train (tokens) | 33.82 / 48 |

## Training

Run the stages in order:

```bash
python -m scripts.train --model sft
python -m scripts.train --model reward
python -m scripts.train --model ppo
python -m scripts.train --model dpo
```

- PPO and DPO both start from `checkpoints/sft`; PPO also needs `checkpoints/reward`.
- Each run saves its checkpoint into `checkpoints/<stage>/`, adds one row to `results/training_stats.csv` and writes a curve into `results/figures/`.
- `python -m scripts.local_train --model sft` is a thin wrapper around `scripts/train.py`.

### Modal

`scripts/modal_train.py` runs the same stages on a Modal GPU. It keeps `checkpoints/` and `results/` in the persistent volume `llm-alignment-from-scratch` and, after each run, downloads the stage checkpoint plus the whole `results/` folder.

```bash
modal setup
python -m scripts.modal_train --model sft
python -m scripts.modal_train --model reward
python -m scripts.modal_train --model ppo
python -m scripts.modal_train --model dpo
```

Run the evaluation on Modal with:

```bash
python -m scripts.modal_train --evaluate
```

`modal run scripts/modal_train.py --model sft` works the same way.

## Evaluation

```bash
python -m scripts.evaluate
```

It generates one greedy summary per held-out post with SFT, PPO and DPO, scores them with the reward model, and writes `results/final_comparison.csv`, `results/eval_summaries.jsonl` and `results/samples.md`.

| metric | meaning |
| --- | --- |
| Mean rϕ | mean reward model score of the generated summaries |
| Win rate vs. SFT | fraction of posts where the policy beats SFT under rϕ, ties count as half a win |
| KL to SFT | KL(π ‖ π_SFT) measured on the policy's own greedy summary |
| Length | generated tokens up to the first EOS |
| ROUGE-L | ROUGE-L against the human reference summary |

## Results

### Training

| Stage | Final train loss | Validation metric | Time / GPU |
| --- | --- | --- | --- |
| SFT | 2.3329 | val. loss: 2.3534 | 0:10:50 / NVIDIA A100-SXM4-40GB |
| Reward model | 0.6081 | pairwise acc.: 0.6210 | 0:20:41 / NVIDIA A100-SXM4-40GB |
| PPO | -0.0041 | final mean rϕ: 6.34 | 0:10:12 / NVIDIA A100-SXM4-40GB |
| DPO | 0.6146 | implicit reward acc.: 0.6130 | 0:35:12 / NVIDIA A100-SXM4-40GB |

### Final comparison

| Policy | Mean rϕ | Win rate vs. SFT | KL to SFT | Length | ROUGE-L |
| --- | --- | --- | --- | --- | --- |
| SFT | 4.8045 | - | 0 | 29.99 | 0.2647 |
| PPO | 6.1958 | 0.8767 | 12.4485 | 31.71 | 0.2301 |
| DPO | 5.7913 | 0.8233 | 3.9720 | 37.66 | 0.2461 |

Both PPO and DPO improve the reward score and win against SFT on the majority of posts, while DPO stays much closer to the SFT policy in KL.

### Files

```
results/
├── data_stats.csv                   # dataset sizes and token length statistics
├── training_stats.csv               # one row per trained stage
├── final_comparison.csv             # metrics for SFT, PPO and DPO
├── eval_summaries.jsonl             # one JSON object per post with reference + three summaries
├── samples.md                       # 10 side-by-side summaries with rϕ scores
├── human_eval.csv                   # empty template for the two human raters
└── figures/
    ├── sft_curves.png
    ├── rwd_curves.png
    ├── ppo_curves.png
    └── dpo_curves.png
```

`human_eval.csv` is an empty template (two raters, SFT/PPO/DPO picks, agreement out of 30) and is filled in by hand, so it is not part of the automatic comparison above.

## Push checkpoints to Hugging Face

```bash
python -m scripts.push_to_hub
```

Uploads the contents of `checkpoints/` to the model repo `<your-username>/llm-alignment-from-scratch`. The repo root contains `sft/`, `reward/`, `ppo/` and `dpo/` directly, without a `checkpoints/` folder.

---
[HuggingFace Checkpoins](https://huggingface.co/Muhammad-Huzzaifa/llm-alignment-from-scratch)

