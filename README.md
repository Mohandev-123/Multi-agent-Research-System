# AI Research Assistant

A local multi-agent research system with a small React web UI.

The pipeline runs four agents:

1. **Search** — generates queries and finds sources with Tavily.
2. **Reader** — scrapes the top pages and extracts evidence.
3. **Writer** — writes a cited research report with the local LLM.
4. **Critic** — scores the report and sends it back for revisions.

The web UI shows live agent statuses, the final report, clickable sources, critic feedback, and a downloadable Markdown report.

## Prerequisites

- Python 3.11+
- Node.js 18+
- [LM Studio](https://lmstudio.ai/) running a server on `http://localhost:1235/v1` with the `qwen3.5-2b` model loaded
- A Tavily API key in `.env`:

```dotenv
TAVILY_API_KEY="tvly-..."
```

The model configuration lives in `agents.py` and is shared by the writer and critic agents.

## Setup

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
cd web
npm install
cd ..
```

## Run the web app

Start the backend in the first PowerShell terminal:

```powershell
.\.venv\Scripts\python.exe -m uvicorn server:app --reload --port 8000
```

Start the frontend in a second PowerShell terminal:

```powershell
cd web
npm run dev
```

Open <http://localhost:5173>.

Enter a topic and click **Start Research**. The progress panel is driven by real events from the pipeline: Search, Reader, Writer, and Critic update as each stage runs. The report only appears after the job completes.

The final report is also saved to `output/` by the existing pipeline.

## Run from the command line

The original CLI still works:

```powershell
.\.venv\Scripts\python.exe pipeline.py
```

Or run the non-interactive example:

```powershell
.\.venv\Scripts\python.exe run_test.py
```

## Tests

Unit tests (no LM Studio or internet required):

```powershell
.\.venv\Scripts\python.exe -m pytest -m "not integration"
```

Live integration test (requires LM Studio, Tavily, and internet):

```powershell
.\.venv\Scripts\python.exe -m pytest -m integration -v
```

## Build the frontend

```powershell
cd web
npm run build
```

The build output is written to `web/dist/`.

## Project structure

```text
agents.py        Search, Reader, Writer, and Critic agents
models.py        Pydantic data models
pipeline.py      Agent orchestration, revision loop, progress events
tools.py         Tavily search and URL scraping tools
server.py        FastAPI backend (non-blocking jobs + report download)
test_pipeline.py Unit and integration tests
web/             React + Vite + Tailwind CSS frontend
output/          Saved Markdown reports (generated)
```

## API

| Method | Endpoint                      | Purpose                              |
| ------ | ----------------------------- | ------------------------------------ |
| POST   | `/api/research`               | Start a job with `{"topic": "..."}`  |
| GET    | `/api/jobs/{job_id}`          | Live status, sources, report, critic |
| GET    | `/api/jobs/{job_id}/report`   | Download the saved Markdown report   |

Jobs run in a background thread, so the start request returns immediately. Job state is kept in memory and the API is intended for local, single-user use.
