import os
import json
import hashlib
import re
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from flask_cors import CORS
import requests

# Optional imports with graceful fallback
try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    from docx import Document
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
CHUNK_SIZE = 1000        # characters per chunk
CHUNK_OVERLAP = 200      # overlap between chunks
TOP_K = 5                # number of chunks to retrieve

# ─── In-memory document store ────────────────────────────────────────────────
# Structure: { file_id: { "filename": str, "chunks": [str], "embeddings": [[float]] } }
document_store: dict = {}


# ─── Text utilities ──────────────────────────────────────────────────────────

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return [c for c in chunks if c.strip()]


def extract_text_from_pdf(path: str) -> str:
    if not PDF_SUPPORT:
        return "[PDF support unavailable. Install PyMuPDF: pip install pymupdf]"
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)


def extract_text_from_docx(path: str) -> str:
    if not DOCX_SUPPORT:
        return "[DOCX support unavailable. Install python-docx: pip install python-docx]"
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)


def extract_text(path: str, filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        return extract_text_from_pdf(path)
    elif ext in ("docx", "doc"):
        return extract_text_from_docx(path)
    elif ext == "txt":
        with open(path, "r", errors="replace") as f:
            return f.read()
    return ""


# ─── Ollama helpers ──────────────────────────────────────────────────────────

def ollama_embed(text: str, model: str = "nomic-embed-text") -> list[float] | None:
    """Get embedding from Ollama."""
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=60,
        )
        if resp.status_code == 200:
            return resp.json().get("embedding")
    except Exception:
        pass
    return None


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x ** 2 for x in a) ** 0.5
    mag_b = sum(x ** 2 for x in b) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def retrieve_chunks(query: str, file_ids: list[str], embed_model: str) -> list[str]:
    """Retrieve top-K most relevant chunks using embeddings or keyword fallback."""
    query_emb = ollama_embed(query, embed_model)

    scored: list[tuple[float, str]] = []

    for fid in file_ids:
        doc = document_store.get(fid)
        if not doc:
            continue
        chunks = doc["chunks"]
        embeddings = doc.get("embeddings", [])

        if query_emb and embeddings:
            for chunk, emb in zip(chunks, embeddings):
                if emb:
                    scored.append((cosine_similarity(query_emb, emb), chunk))
        else:
            # Keyword fallback
            q_lower = query.lower()
            for chunk in chunks:
                score = sum(1 for w in q_lower.split() if w in chunk.lower())
                scored.append((float(score), chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:TOP_K]]


def list_ollama_models() -> list[str]:
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        if resp.status_code == 200:
            return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        pass
    return []


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        ollama_ok = resp.status_code == 200
    except Exception:
        ollama_ok = False
    return jsonify({
        "status": "ok",
        "ollama_connected": ollama_ok,
        "pdf_support": PDF_SUPPORT,
        "docx_support": DOCX_SUPPORT,
    })


@app.route("/api/models")
def models():
    return jsonify({"models": list_ollama_models()})


@app.route("/api/upload", methods=["POST"])
def upload():
    files = request.files.getlist("files")
    embed_model = request.form.get("embed_model", "nomic-embed-text")
    results = []

    for f in files:
        if not f.filename:
            continue

        filename = f.filename
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in ("pdf", "docx", "doc", "txt"):
            results.append({"filename": filename, "status": "error", "message": "Unsupported file type"})
            continue

        file_id = hashlib.md5(filename.encode()).hexdigest()[:8]
        save_path = os.path.join(UPLOAD_FOLDER, f"{file_id}_{filename}")
        f.save(save_path)

        text = extract_text(save_path, filename)
        if not text.strip():
            results.append({"filename": filename, "status": "error", "message": "Could not extract text"})
            continue

        chunks = chunk_text(text)

        # Try to embed each chunk (may fail if model not available)
        embeddings = []
        for chunk in chunks:
            emb = ollama_embed(chunk, embed_model)
            embeddings.append(emb)

        document_store[file_id] = {
            "filename": filename,
            "chunks": chunks,
            "embeddings": embeddings,
            "char_count": len(text),
        }

        embedded_count = sum(1 for e in embeddings if e)
        results.append({
            "filename": filename,
            "file_id": file_id,
            "status": "ok",
            "chunks": len(chunks),
            "embedded": embedded_count,
            "chars": len(text),
        })

    return jsonify({"results": results, "total_docs": len(document_store)})


@app.route("/api/documents")
def list_documents():
    docs = [
        {
            "file_id": fid,
            "filename": d["filename"],
            "chunks": len(d["chunks"]),
            "chars": d["char_count"],
        }
        for fid, d in document_store.items()
    ]
    return jsonify({"documents": docs})


@app.route("/api/documents/<file_id>", methods=["DELETE"])
def delete_document(file_id):
    if file_id in document_store:
        del document_store[file_id]
        return jsonify({"status": "deleted"})
    return jsonify({"status": "not_found"}), 404


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json or {}
    question = data.get("question", "").strip()
    file_ids = data.get("file_ids", list(document_store.keys()))
    chat_model = data.get("chat_model", "llama3")
    embed_model = data.get("embed_model", "nomic-embed-text")
    history = data.get("history", [])

    if not question:
        return jsonify({"error": "No question provided"}), 400
    if not document_store:
        return jsonify({"error": "No documents uploaded yet"}), 400

    chunks = retrieve_chunks(question, file_ids, embed_model)
    context = "\n\n---\n\n".join(chunks)

    system_prompt = (
        "You are a helpful document assistant. "
        "Answer the user's question using ONLY the provided document context. "
        "If the answer is not in the context, say so clearly. "
        "Be concise and accurate.\n\n"
        f"DOCUMENT CONTEXT:\n{context}"
    )

    messages = [{"role": "system", "content": system_prompt}]
    for h in history[-6:]:  # last 3 turns
        messages.append(h)
    messages.append({"role": "user", "content": question})

    def generate():
        try:
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={"model": chat_model, "messages": messages, "stream": True},
                stream=True,
                timeout=120,
            )
            for line in resp.iter_lines():
                if line:
                    try:
                        obj = json.loads(line)
                        token = obj.get("message", {}).get("content", "")
                        if token:
                            yield f"data: {json.dumps({'token': token})}\n\n"
                        if obj.get("done"):
                            yield f"data: {json.dumps({'done': True})}\n\n"
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("✅  Ollama RAG server starting on http://localhost:5000")
    print(f"   Ollama URL : {OLLAMA_BASE_URL}")
    print(f"   PDF support : {PDF_SUPPORT}")
    print(f"   DOCX support: {DOCX_SUPPORT}")
    app.run(debug=True, port=5000)
