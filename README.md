# DocMind — Local AI Document Chat

A fully local RAG (Retrieval-Augmented Generation) application that lets you upload PDF and Word documents and chat with them using Ollama models. Your data never leaves your machine.

---

## What It Does

Upload one or more documents (PDF, DOCX, TXT), then ask questions in plain English. The backend splits your files into chunks, embeds them with an Ollama embedding model, retrieves the most relevant passages for each question, and feeds them to a local LLM as context.

---

## Requirements

- **Python 3.10+**
- **[Ollama](https://ollama.com/)** running locally on port `11434`
- At minimum one chat model and one embedding model pulled in Ollama

---

## Ollama Setup

Install Ollama from https://ollama.com, then pull the models you want:

```bash
# Chat models (pick at least one)
ollama pull llama3
ollama pull mistral
ollama pull gemma

# Embedding model (required for semantic search)
ollama pull nomic-embed-text
```

Confirm Ollama is running:
```bash
ollama serve          # or it auto-starts as a background service
curl http://localhost:11434/api/tags
```

---

## Installation

```bash
git clone <repo-url>
cd ollama-rag

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Running

```bash
python app.py
```

Open your browser at **http://localhost:5000**

---

## Usage

1. **Upload documents** — drag and drop files into the sidebar or click Browse. Supports PDF, DOCX, DOC, and TXT.
2. **Pick models** — select your chat model and embedding model from the dropdowns.
3. **Ask questions** — type any question about your documents and press Enter. Answers stream back in real time.
4. **Manage documents** — uncheck a document to exclude it from a query, or click ✕ to remove it.
5. **Clear chat** — click "↺ Clear" to start a fresh conversation (documents remain loaded).

------------

## Project Structure

```
ollama-rag/
├── app.py              # Flask backend — file parsing, RAG pipeline, Ollama API
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html      # Single-file frontend (HTML + CSS + JS)
└── uploads/            # Uploaded files stored here (created automatically)
```

---------------------

## Configuration

You can customize these values at the top of `app.py`:

| Variable        | Default              | Description                              |
|-----------------|----------------------|------------------------------------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint                  |
| `CHUNK_SIZE`    | `1000`               | Characters per text chunk                |
| `CHUNK_OVERLAP` | `200`                | Overlap between consecutive chunks       |
| `TOP_K`         | `5`                  | Number of chunks retrieved per question  |

To use a remote Ollama instance:
```bash
OLLAMA_BASE_URL=http://192.168.1.10:11434 python app.py
```

-----------------

## How It Works

```
Upload PDF/DOCX
     ↓
Extract full text
     ↓
Split into overlapping chunks
     ↓
Embed each chunk via Ollama (nomic-embed-text)
     ↓
User asks a question
     ↓
Embed the question
     ↓
Cosine similarity → top-K chunks retrieved
     ↓
Chunks injected into LLM prompt as context
     ↓
Ollama streams the answer back
```

If the embedding model is not available, the app falls back to keyword-based retrieval automatically.

---

## API Endpoints

| Method | Path                    | Description                        |
|--------|-------------------------|------------------------------------|
| GET    | `/api/health`           | Check Ollama connection and deps   |
| GET    | `/api/models`           | List available Ollama models       |
| POST   | `/api/upload`           | Upload and index documents         |
| GET    | `/api/documents`        | List all loaded documents          |
| DELETE | `/api/documents/<id>`   | Remove a document                  |
| POST   | `/api/chat`             | Ask a question (SSE streaming)     |

---

## Troubleshooting

**"Ollama ✗" in the header**
Ollama is not running or not reachable. Start it with `ollama serve`.

**"Could not extract text" on upload**
The file may be a scanned image-based PDF. OCR is not included — use a text-based PDF.

**Slow responses**
This depends on your hardware and model size. Smaller models like `phi3` or `gemma:2b` are faster on CPU-only machines.

**Embeddings not working / keyword fallback**
Run `ollama pull nomic-embed-text` and make sure Ollama is running before uploading files.

--------------------

## Dependencies

| Package         | Purpose                          |
|-----------------|----------------------------------|
| `flask`         | Web server                       |
| `flask-cors`    | Cross-origin request handling    |
| `requests`      | HTTP calls to Ollama API         |
| `pymupdf`       | PDF text extraction              |
| `python-docx`   | DOCX text extraction             |
