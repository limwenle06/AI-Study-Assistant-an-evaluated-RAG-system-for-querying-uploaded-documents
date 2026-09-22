# AI Study Assistant
A Streamlit study assistant that answers questions about a text-based PDF using retrieval-augmented generation (RAG). It finds relevant passages in the document, asks an OpenAI model to answer from those passages, and shows the source pages alongside the answer.

## What it does

- Upload and process a PDF, then ask questions about it.
- Show the document passages used for each answer, including page and chunk numbers.
- Save processed documents in a local SQLite database so they can be selected again from the **PDFs** page without re-uploading or re-embedding them.
- Detect an already-processed PDF by its file contents, not just its filename.

## How it works

**When you process a new PDF:**

1. Extract text page by page using `pypdf`'s layout extraction mode.
2. Split each page into chunks of up to 200 words, with 40 words of overlap between neighboring chunks on that page.
3. Send the chunks to OpenAI's `text-embedding-3-small` model to create embeddings (numeric representations of the text).
4. Save the PDF's filename, content hash, chunk text, page numbers, and embeddings in `data/study_assistant.db`.

**When you ask a question:**

1. Embed the question with the same embedding model.
2. Compare its embedding with the saved chunk embeddings using cosine similarity and select the three highest-scoring chunks.
3. Send the question and those chunks to the answer model (`gpt-5.6-luna`) with instructions to answer from the supplied sources and cite them.
4. Display the generated answer and expandable source passages.

Selecting a saved PDF from the **PDFs** page loads its chunks from SQLite. It does not need another PDF embedding request, but asking a new question still calls the API for the question embedding and answer.

## Run locally

You need Python, an OpenAI API key with available API credits, and an internet connection for API requests. API usage may incur charges. Run these commands from the project folder.

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run main.py
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run main.py
```

Before running the app, provide `OPENAI_API_KEY` as an environment variable. The app also calls `load_dotenv()`, so a local `.env` file can supply it. Keep your key private and do not commit it to Git.

Streamlit prints a local URL in the terminal. Open that URL in your browser, upload a PDF on **Home**, and select **Process PDF**. After processing, you can ask questions or switch to a previously saved document from **PDFs**.

## Data and API use

The uploaded PDF is copied to a temporary file for processing and that copy is removed afterward. The local SQLite database keeps extracted text and embeddings, so treat it as document data and do not commit it if your PDFs are private. Creating document embeddings sends the extracted chunks to OpenAI; asking a question sends the question for embedding and sends the question plus retrieved chunks to the answer model.

## Run tests

```bash
python -m pytest
```

The automated tests use fake API clients and do not make paid OpenAI requests. They cover PDF text extraction, chunking, embeddings, retrieval, answer generation, SQLite storage, and the pipeline.

## Current limitations

- The app extracts written PDF text. Scanned pages, diagrams, and image-heavy lecture slides are not reliably understood; use text-based PDFs for best results.
- PDF extraction can lose some layout or reading order, particularly in complex documents.
- Retrieval uses the three highest-scoring chunks without a minimum relevance threshold. A citation shows which chunk was supplied, but it does not guarantee every statement in the answer is supported by the PDF. Check important answers against the displayed sources.
- Previous messages are shown during the current session, but they are **not** passed to the model. Follow-up questions need to stand on their own, and chat history is not saved between sessions.
- Only one PDF is active for questions at a time. The local database stores processed text and embeddings, not a permanent copy of the original PDF.
- This is a document study tool, not a crisis-support or professional-advice service.

## Project structure

| Path | Purpose |
| --- | --- |
| `main.py` | Streamlit interface and session state |
| `src/document_processing.py` | Page-aware PDF text extraction |
| `src/chunking.py` | Overlapping, page-aware text chunks |
| `src/embeddings.py` | OpenAI embeddings for chunks and questions |
| `src/retrieval.py` | Cosine-similarity scoring and top-three retrieval |
| `src/answer_generation.py` | Source formatting and answer generation |
| `src/database.py` | Local SQLite storage and document lookup |
| `src/pipeline.py` | Connects processing, retrieval, and answering |
| `tests/` | Offline automated tests |

Built by Lim Wen Le as a portfolio and learning project.
