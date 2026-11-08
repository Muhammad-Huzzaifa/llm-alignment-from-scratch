# llm-alignment-from-scratch
custom PyTorch implementation of the full LLM alignment pipeline: SFT → reward model → PPO → DPO built from scratch on SmolLM2 on task of learning to summarize from human feedback

```
├── app
│   ├── inference.py
│   ├── main.py
│   ├── static
│   │   ├── script.js
│   │   └── style.css
│   └── templates
│       └── index.html
├── checkpoints
├── notebooks
│   └── experiments.ipynb
├── README.md
├── requirements.txt
├── results
├── scripts
│   ├── evaluate.py
│   ├── local_train.py
│   ├── modal_train.py
│   ├── push_to_hub.py
│   └── train.py
└── src
    ├── batching.py
    ├── common.py
    ├── config.py
    ├── data.py
    ├── dpo.py
    ├── evaluation.py
    ├── models.py
    ├── ppo.py
    ├── reward.py
    ├── sft.py
    └── utils.py
```

```
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
