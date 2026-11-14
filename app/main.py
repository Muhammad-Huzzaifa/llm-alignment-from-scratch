import csv
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))  # make "app" and "src" importable when this file is run directly

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.inference import summarize
from src.config import DEVICE, MAX_PROMPT_TOKENS, RESULTS_DIR

APP_DIR = Path(__file__).resolve().parent

app = FastAPI(title="llm-alignment-from-scratch")
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
templates = Jinja2Templates(directory=APP_DIR / "templates")


class SummarizeRequest(BaseModel):
    post: str = Field(min_length=1, max_length=20000)
    methods: list[str] = ["sft", "ppo", "dpo"]


def load_metrics():
    """read the offline comparison from results/final_comparison.csv for the metrics strip

    Returns:
        list: one dict per policy, empty when the evaluation has not been run yet
    """
    path = Path(RESULTS_DIR) / "final_comparison.csv"
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return [
            {
                "name": row["Policy"],
                "reward": row["Mean rϕ"],
                "win": row.get("Win rate vs. SFT", "-"),
                "kl": row.get("KL to SFT", "-"),
                "rouge": row.get("ROUGE-L", "-"),
            }
            for row in csv.DictReader(f)
        ]


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    """render the single page app with the offline metrics when they exist

    Args:
        request (Request): the incoming request

    Returns:
        TemplateResponse: the rendered index page
    """
    return templates.TemplateResponse(
        request, "index.html",
        {"metrics": load_metrics(), "device": DEVICE, "max_prompt_tokens": MAX_PROMPT_TOKENS},
    )


@app.post("/api/summarize")
def api_summarize(body: SummarizeRequest):
    """generate and score summaries for the posted text

    Args:
        body (SummarizeRequest): the post text and the selected methods

    Returns:
        dict: one result per method with the summary and its reward
    """
    try:
        results = summarize(body.post, body.methods)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"generation failed: {e}")
    return {"results": results}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
