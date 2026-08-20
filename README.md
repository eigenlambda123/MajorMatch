# MajorMatch

AI-powered course + career explorer for students, advisors, and incoming freshmen.  
Ask natural-language questions about majors, courses, salaries, and career fit; MajorMatch replies conversationally and uses grounded tools when helpful (semantic course search, career-context lookup, and a career-track predictor).

## Highlights
- Chat-first Streamlit UI with tool-grounded replies.
- Semantic course search with embeddings and 2D projections (PCA / UMAP / t-SNE).
- Career-context tool (Adzuna) for live job-market data (optional).
- Simple ML track predictor with an interactive feature-selection UI.
- Portable fallback storage for embeddings (Postgres float[]); pgvector optional.

## Stack
- Language(s): Jupyter Notebooks (project artifacts) + Python 3.10+
- Runtime / framework: Streamlit app (streamlit_app.py)
- Notable libraries: sentence-transformers, scikit-learn, SQLAlchemy, plotly, streamlit

## Repository layout (top-level)
```
README.md                <- this file
streamlit_app.py         <- Streamlit UI and main app loop
app_logic.py             <- high-level UI helpers and orchestrator helpers
course_index.py          <- course index, embeddings, projection & DB layer
requirements.txt         <- Python dependencies for the main app
api/                     <- tool implementations and orchestrator logic
  ollama.py              <- transport to Ollama (chat + streaming)
  orchestrator.py        <- tool-calling orchestrator & tool schemas
  predict.py             <- prediction helper and feature listing
  search.py              <- higher-level search or wrapper utilities
data/                    <- place course CSV(s) here (title, description)
docs/                    <- architecture notes, setup guide, implementation log
scripts/                 <- utility scripts (indexing/embed.py)
tests/                   <- lightweight pytest checks
ml_model/                <- model-training artifacts & separate requirements
```

## Quick start (minimum)
1. Clone the repo:
   ```bash
   git clone https://github.com/eigenlambda123/MajorMatch.git
   cd MajorMatch
   ```

2. Create and activate a Python virtual environment:
   - Linux / macOS
     ```bash
     python -m venv venv
     source venv/bin/activate
     ```
   - Windows (PowerShell)
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Prepare PostgreSQL (recommended) or use the default local connection string:
   - Start Postgres and create a DB + user:
     ```sql
     CREATE DATABASE semantic_search;
     CREATE USER postgres WITH PASSWORD 'postgres';
     GRANT ALL PRIVILEGES ON DATABASE semantic_search TO postgres;
     ```
   - Set DATABASE_URL env var (example):
     - Linux / macOS
       ```bash
       export DATABASE_URL="postgresql+psycopg2://postgres:postgres@localhost:5432/semantic_search"
       ```
     - Windows (PowerShell)
       ```powershell
       $env:DATABASE_URL = "postgresql+psycopg2://postgres:postgres@localhost:5432/semantic_search"
       ```

   Note: MajorMatch attempts to create the `vector` extension if available. `pgvector` is optional — embeddings are stored as float[] by default for portability.

5. (Optional) Start Ollama for local model serving (Chat + tool-calling):
   ```bash
   ollama serve
   ```
   Environment variables:
   - OLLAMA_BASE_URL — defaults to `http://localhost:11434`
   - OLLAMA_MODEL — defaults to `llama2:latest` (you may prefer a smaller/faster local model)

6. Build the course index (reads CSVs from `data/` and writes embeddings/projections to Postgres):
   - Default: index all CSVs under `data/` or the specific file:
     ```bash
     python scripts/embed.py
     # or point at a file:
     python scripts/embed.py data/courses.csv
     ```
   - CSV requirements: must include `title` and `description` columns; rows missing them are skipped.

7. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```
   Open the URL printed by Streamlit (usually http://localhost:8501).

## Running tests
Run the core test suite (example):
- Linux / macOS:
  ```bash
  PYTHONPATH='.' pytest -q
  ```
- Windows (PowerShell):
  ```powershell
  $env:PYTHONPATH='.'; .\venv\Scripts\python -m pytest -q
  ```

There is a focused test for orchestrator behavior: `tests/test_orchestrator.py`.

## Environment variables (summary)
- DATABASE_URL — SQLAlchemy-compatible DB URL (see examples above).
- EMBEDDING_MODEL — sentence-transformers model name (default: `all-MiniLM-L6-v2`).
- EMBEDDING_DIMENSION — dimension of embeddings (default: 384).
- OLLAMA_BASE_URL — Ollama server URL (default: `http://localhost:11434`).
- OLLAMA_MODEL — Ollama model name to use for chat.
- ADZUNA_APP_ID / ADZUNA_APP_KEY — optional credentials for career-context job-market data.

## Data & CSV format
- Put one or more CSV files into the `data/` directory.
- Required columns: `title`, `description`.
- The indexer computes sentence-transformer embeddings and stores them in Postgres (float[] by default). Projections (PCA/UMAP/t-SNE) are computed and persisted.

## How the pieces fit together (short)
- streamlit_app.py implements the chat-first UI. User messages are sent to the orchestrator which decides whether to call tools.
- api/orchestrator.py handles tool-calling flow, executes tools (predict, career-context, semantic search), and asks the chat model to produce a grounded final reply.
- course_index.py stores and queries the course corpus (embeddings + projection coordinates) in Postgres and provides the semantic search + projection helpers.
- api/ollama.py handles model calls (streaming and batched) to the locally hosted Ollama model.

## Troubleshooting & tips
- If embeddings fail due to missing model packages:
  - Ensure `sentence-transformers`, `torch` and the model weights are installed. Large models can require extra disk and memory.
- If Postgres extension `vector` creation fails: it’s okay — the app will fall back to float[] storage and compute similarity in Python.
- If Ollama is unreachable: set `OLLAMA_BASE_URL` correctly or start Ollama locally. Without Ollama, the assistant can’t perform tool calls; the UI still loads.
- If UMAP / t-SNE projection raises errors on very small corpora, try indexing a larger set or use PCA as a fallback.

## Development notes
- The prediction tool expects explicit feature names. The UI can open an interactive prediction selector when the model requests it.
- To reindex after updating `data/`, run `python scripts/embed.py` again; the indexer clears and rebuilds the table.
- Course projection methods are available: PCA, UMAP, t-SNE. Computation happens at index time and uses cached course embeddings for fast queries.

## Contributing
- Add issues for bugs or feature requests.
- Keep new CSVs under `data/`; follow existing CSV column names.
- If adding heavy dependencies, update `requirements.txt` and note the change in `docs/IMPLEMENTATION.md`.

## Contact / Notes
For architecture, data flow, and implementation details see:
- docs/PROJECT_DOCUMENTATION.md
- docs/instruction.md
- docs/IMPLEMENTATION.md
