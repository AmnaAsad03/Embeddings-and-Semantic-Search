# Medical Knowledge Assistant — Implementation Report

This repository contains my Week 6 internship work, focused on building a medical information service using Retrieval-Augmented Generation (RAG), hybrid search, evidence-based answers, citations, and safety checks.

## Project summary

This repository implements a small FastAPI service for searching a local collection of medical reference documents and generating answers from retrieved passages. Search combines semantic retrieval with a BM25 keyword search. The answer endpoint applies question checks, retrieves passages, asks a Groq-hosted language model to answer from those passages, and checks the generated text for safety concerns.

The current document collection contains 12 Markdown files in `data/documents/`. The files cover selected topics including asthma, diabetes, pulmonary hypertension, GERD, migraine, anemia, cardiovascular disease, pneumonia, influenza, food poisoning, depression, and medication safety. This is a limited demonstration collection, not a comprehensive medical knowledge base.

## Run it locally

Use Python 3.10 or newer. In PowerShell, from the repository root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Answer generation requires a Groq API key. Create a `.env` file in the repository root:

```text
GROQ_API_KEY=your_groq_api_key
```

Do not commit the key. The application loads it from the environment when the generation module is imported.

## Build or rebuild the local search index

Run these commands from the repository root, in order:

```powershell
python -m src.ingestion.ingest_documents
python -m src.chunking.create_chunks
python -m src.embedding.create_embeddings
python -m src.retrieval.create_vector_db
```

What each step does:

1. `ingest_documents` reads `data/documents/*.md`, cleans whitespace, checks required metadata, and writes cleaned files and an ingestion report under `data/processed/`.
2. `create_chunks` reads the cleaned files and writes `data/processed/chunks/chunks.json`. The defaults in `configs/rag_config.yaml` are recursive chunking, 800 characters per chunk, and 120 characters of overlap.
3. `create_embeddings` embeds the chunks with `sentence-transformers/all-MiniLM-L6-v2` and writes `data/processed/embeddings/embeddings.npy`.
4. `create_vector_db` loads the chunks and embeddings and upserts them into the persistent Chroma collection `medical_knowledge_base` under `data/vector_db/`.

The processed files, embedding matrix, and vector database are generated locally and are not part of the source dataset. Run the build steps again after a fresh clone or after changing the documents. The embedding model may need to be downloaded the first time it runs.

## Start the API

Once the index has been built:

```powershell
python main.py
```

The development server listens on `http://127.0.0.1:8000`. FastAPI's interactive documentation is at `http://127.0.0.1:8000/docs`.

### Endpoints

| Method | Path | Request | Response |
| --- | --- | --- | --- |
| `GET` | `/health` | None | Service status and version |
| `POST` | `/search` | `{"query": "...", "top_k": 5}` | JSON containing ranked passages, metadata, and vector/BM25/hybrid scores |
| `POST` | `/ask` | `{"question": "...", "top_k": 5}` | Plain-text streamed answer; citations are included in the answer text when evidence is sufficient |

`top_k` defaults to 5 and must be between 1 and 20. For example, a `/search` request can be sent with:

```json
{
  "query": "What are common symptoms of asthma?",
  "top_k": 5
}
```

`/ask` uses the `question` field instead of `query`. It returns plain text rather than the structured `AskResponse` schema declared in `src/api/schemas.py`.

## How a request is handled

### Search

`src/retrieval/hybrid_search.py` asks both searchers for up to `top_k` passages:

- `VectorSearcher` encodes the query with `all-MiniLM-L6-v2` and queries the Chroma collection.
- `BM25Searcher` scores the chunks in `data/processed/chunks/chunks.json` using token matches.
- Each result set is normalized separately. The current hybrid score is an even blend: 50% vector score and 50% BM25 score. Results are sorted by that score.

The `/search` endpoint returns passage text, metadata, rank, and the component and combined scores. It is useful for checking whether the expected source passages appear before testing generated answers.

### Answers

`src/generation/rag_pipeline.py` handles `/ask` requests in this order:

1. `QuestionValidator` classifies the question and can allow it, refuse it, or return an emergency escalation message.
2. `HybridSearcher` retrieves passages.
3. `MedicalAnswerGenerator` checks whether any result meets the configured minimum hybrid score (`0.15`), selects up to five passages, and sends them with the question to Groq.
4. Unique document citations are collected from retrieved metadata.
5. `MedicalSafetyChecker` scans the generated answer. It can allow an answer, add a warning, refuse it, or escalate it.
6. If evidence is insufficient, the generator returns an abstention message instead of calling the model.

The model is configured in `src/generation/config.py` as `openai/gpt-oss-20b`, with temperature `0` and a maximum of 800 tokens. The prompts instruct it to use only the retrieved evidence. These instructions and the safety checks reduce risk but do not prove that an answer is complete or medically correct.

## Tests and evaluation

Run the automated tests from the repository root:

```powershell
pytest
```

To run one test file, for example:

```powershell
pytest tests/test_question_validator.py
```

The test suite includes API, health, question-validation, and safety-related tests. Some integration tests or evaluation scripts may need the locally built chunks and Chroma collection. Tests that exercise real answer generation also need a working `GROQ_API_KEY`.

The repository includes golden retrieval and answer datasets in `data/evaluation/`. After building the index, run retrieval evaluation with:

```powershell
python -m src.evaluation.evaluate_golden_retrieval
```

This compares retrieved chunk IDs against the expected chunk IDs in `golden_retrieval.json` and reports retrieval metrics. Additional scripts under `src/evaluation/` evaluate the answer pipeline and safety behavior; check their inputs and API-key requirements before running them. Evaluation results depend on the current documents, generated index, model, and dataset, so numbers from one run should not be treated as permanent project guarantees.

## Repository map

```text
configs/rag_config.yaml       Chunking settings and output paths
data/documents/               Source medical Markdown files
data/evaluation/              Golden retrieval and answer datasets
src/ingestion/                Document cleaning and validation
src/chunking/                 Chunk creation strategies
src/embedding/                Sentence-transformer embedding generation
src/retrieval/                Chroma vector, BM25, and hybrid search
src/generation/               Prompts, Groq client, citations, and RAG pipeline
src/safety/                   Question validation and answer safety checks
src/api/                      FastAPI application, schemas, and routes
src/evaluation/               Retrieval, generation, and safety evaluations
tests/                        Pytest API and validator tests
main.py                       Local Uvicorn entry point
requirements.txt              Python dependencies
```

## Limitations 

- The indexed knowledge is limited to the documents in `data/documents/`. Missing or outdated information cannot be retrieved.
- Question validation and answer safety checks are rule-based code, not clinical review or a medical-grade safety guarantee.
- Retrieval scores are ranking signals; a high score does not establish that a source is authoritative or that the answer is correct.
- `/ask` depends on an external Groq service and may be unavailable without network access or a valid API key. `/search` uses the locally built index.
- Do not use this project to diagnose a condition, choose or change treatment, or handle an emergency. Users should consult a qualified clinician; emergencies require urgent local care.
