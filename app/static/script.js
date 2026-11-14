const form = document.getElementById("compose-form");
const postInput = document.getElementById("post");
const charCount = document.getElementById("char-count");
const grid = document.getElementById("results-grid");
const statusLine = document.getElementById("status");
const generateButton = document.getElementById("generate");

const LABELS = { sft: "SFT", ppo: "PPO", dpo: "DPO" };

function updateCount() {
  const n = postInput.value.length;
  charCount.textContent = `${n} character${n === 1 ? "" : "s"}`;
}

function setStatus(message, isError = false) {
  statusLine.textContent = message;
  statusLine.classList.toggle("is-error", isError);
}

function skeleton(method) {
  return `
    <article class="result-card is-loading" data-method="${method}">
      <header class="result-head">
        <span class="result-method">${LABELS[method]}</span>
        <span class="result-reward">rϕ …</span>
      </header>
      <div class="skeleton-lines"><span></span><span></span><span></span><span></span></div>
      <footer class="result-foot"><span>loading model</span><span></span></footer>
    </article>`;
}

function showSkeletons(methods) {
  grid.innerHTML = methods.map(skeleton).join("");
}

function renderResults(results) {
  grid.innerHTML = results.map((r) => `
    <article class="result-card" data-method="${r.method}">
      <header class="result-head">
        <span class="result-method">${LABELS[r.method] || r.method}</span>
        <span class="result-reward" title="reward model score">rϕ ${Number(r.reward).toFixed(2)}</span>
      </header>
      <p class="result-text"></p>
      <footer class="result-foot"><span>${r.length} tokens</span><span>${r.seconds}s</span></footer>
    </article>`).join("");

  grid.querySelectorAll(".result-card").forEach((card, i) => {
    card.querySelector(".result-text").textContent = results[i].summary;
  });
}

postInput.addEventListener("input", updateCount);
postInput.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    form.requestSubmit();
  }
});
updateCount();

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const post = postInput.value.trim();
  const methods = [...form.querySelectorAll('input[name="method"]:checked')].map((input) => input.value);

  if (!post) {
    setStatus("Write a post first.", true);
    postInput.focus();
    return;
  }
  if (methods.length === 0) {
    setStatus("Pick at least one model.", true);
    return;
  }

  showSkeletons(methods);
  setStatus(`Running ${methods.map((m) => LABELS[m]).join(" → ")} — models load one at a time, so this can take a while.`);
  generateButton.disabled = true;
  generateButton.textContent = "Generating…";

  try {
    const response = await fetch("/api/summarize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ post, methods }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data.detail || `request failed (${response.status})`);
    }

    renderResults(data.results);
    setStatus(`Done — ${data.results.map((r) => `${LABELS[r.method]} ${Number(r.reward).toFixed(2)}`).join(" · ")}`);
    document.getElementById("results").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    grid.innerHTML = "";
    setStatus(error.message, true);
  } finally {
    generateButton.disabled = false;
    generateButton.textContent = "Generate summaries";
  }
});
