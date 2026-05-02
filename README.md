# OpenImpactLab — AI Research System (Project 3)

A three-tier automated research system powered by Google Gemini. Three AI agents collaborate in real time to research any question, with quality control built into the pipeline.

---

## Architecture

```
User (Browser)
     │
     ▼
┌─────────────┐     ┌─────────────┐     ┌──────────────┐
│  DIRECTOR   │────▶│   MANAGER   │────▶│  RESEARCHER  │
│             │     │             │     │              │
│ Breaks the  │     │ Delegates   │     │ Searches the │
│ question    │     │ search      │     │ web via      │
│ into sub-   │     │ queries &   │     │ Google Search│
│ questions,  │     │ reviews     │     │ Grounding,   │
│ synthesizes │     │ quality     │     │ returns cited│
│ final answer│     │ (max 3 iter)│     │ responses    │
└─────────────┘     └─────────────┘     └──────────────┘
```

**Flow:**
1. Director breaks the user's question into 2–4 focused sub-questions
2. For each sub-question, Manager generates an optimized search query
3. Researcher executes the search and returns a grounded, sourced response
4. Manager evaluates quality (A / C / R / Q scores) — if rejected, requests a follow-up search (up to 3 iterations)
5. Director synthesizes all approved results into a final answer

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.9+ |
| Web Framework | Flask |
| LLM / Search | Google Gemini (`gemma-4-31b-it` default) + Google Search Grounding |
| Frontend | Vanilla HTML / CSS / JavaScript (SSE streaming) |
| Config | `python-dotenv` |

---

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd OpenImpactLab-Project-3
```

### 2. Install dependencies

```bash
pip3 install -r requirements.txt
```

Or with a virtual environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemma-4-31b-it
```

> Get a Gemini API key at [aistudio.google.com](https://aistudio.google.com)
>
> **Note:** The Researcher Agent uses Google Search Grounding, which requires a Gemini model that supports it (e.g. `gemini-2.0-flash`). `gemma-4-31b-it` does not support grounding — set `GEMINI_MODEL=gemini-2.0-flash` if you need web search.

### 4. Run

```bash
python3 app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## Project Structure

```
OpenImpactLab-Project-3/
├── app.py                  # Flask server — 3 routes: /, /research, /stream/<id>
├── requirements.txt
├── .env.example
├── src/
│   ├── gemini_client.py    # Shared Gemini client + generate() with retry logic
│   ├── researcher.py       # Researcher Agent — grounded web search
│   ├── manager.py          # Manager Agent — delegation & quality evaluation
│   ├── director.py         # Director Agent — planning & final synthesis
│   └── orchestrator.py     # Pipeline logic + SSE event queue
├── templates/
│   └── index.html          # Single-page UI
└── static/
    └── style.css
```

---

## Manager Quality Scores

The Manager evaluates each Researcher response and displays scores in the format:

```
APPROVED / REJECTED   A:n  C:n  R:n  Q:n
```

| Score | Name | Description |
|---|---|---|
| **A** | Accuracy | Factual correctness; no hallucinations or fabricated claims |
| **C** | Completeness | All key aspects of the sub-question are addressed |
| **R** | Relevance | Answer stays focused on what was asked |
| **Q** | Citation Quality | Multiple web searches were performed and sources are present |

All four scores must reach **4 or above** to be approved. After the 3rd iteration the response is force-approved with the best available answer.

---

## API Routes

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Serves the web UI |
| `POST` | `/research` | Starts a research job; returns `{ request_id }` |
| `GET` | `/stream/<request_id>` | SSE stream of agent log events |

### SSE Event Types

| Event | Source | Data |
|---|---|---|
| `planning` | Director | Sub-question list |
| `delegation` | Manager | Search query |
| `research_start` | Researcher | Query being searched |
| `research_result` | Researcher | Sources retrieved |
| `evaluation` | Manager | Scores + critique |
| `approved` | Manager | Sub-question resolved |
| `rejected` | Manager | Follow-up query |
| `synthesis_start` | Director | Synthesis beginning |
| `done` | System | Final answer + sources |
| `error` | System | Error message |

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | Yes | — | Your Gemini API key |
| `GEMINI_MODEL` | No | `gemma-4-31b-it` | Model used by all three agents |
