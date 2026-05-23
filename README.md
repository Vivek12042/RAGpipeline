# 🎓 B.Tech Course-Book Advanced RAG Assistant

An enterprise-ready **Advanced Retrieval-Augmented Generation (RAG)** pipeline designed to perform semantic and keyword querying over complex, semi-structured documents (like academic course books, curriculum schedules, and credit maps). 

This pipeline is built on **LangChain** and runs **100% free and locally** for document ingestion and embedding generation using CPU-friendly Hugging Face transformers. It features a **resilient multi-provider LLM selector** supporting **Google Gemini, Groq Cloud, OpenAI, and local Ollama** with auto-installing libraries.

---

## 📊 Pipeline Architecture

```
                    1. DATA INGESTION PIPELINE (Offline / Local)
                    
  📄 B.Tech Course-book PDF 
           │
           ▼  (PyPDFLoader)
  🧩 Loaded Pages (137 Pages)
           │
           ▼
  ✂️ Parent-Child Chunking Strategy
     ├── 👨 Parent Chunks (1200 chars) ──> [InMemory JSON Store] ──> 🔎 Sparse Index (BM25)
     └── 👶 Child Chunks (300 chars)   ──> [all-MiniLM-L6-v2]  ──> 🗄️ Dense Index (Chroma DB)
     

                    2. RETRIEVAL & SYNTHESIS PIPELINE (Online / Runtime)
                    
  👤 User Query
       │
       ▼
  🔄 Query Expansion Layer (LLM generates 3 alternate query perspectives for high recall)
       │
       ├──> 🧠 Dense Semantic Search (Chroma DB child vectors mapped back to Parent ID)
       └──> 🔎 Sparse Keyword Search (BM25 search directly on Parent Chunks)
       │
       ▼
  🤝 Reciprocal Rank Fusion (RRF) (Aligns and merges matches from dense and sparse streams)
       │
       ▼
  📝 Citations & Source Page Tracker
       │
       ▼
  🤖 Grounded Prompt Engineering ──> [Chat LLM (Gemini, Groq, OpenAI, Ollama)]
       │
       ▼
  ✨ Final Grounded Response with Page-by-Page Citations (e.g. [Page 45])
```

---

## 🚀 Key Features

* **Parent-Child Chunking**: Splits text into small child chunks for precise semantic matching, but retrieves larger parent paragraphs to provide the LLM with complete contextual understanding.
* **Hybrid Search (Dense + Sparse)**: Integrates semantic vector search (Chroma DB) with classic keyword matching (BM25) to cleanly capture exact course codes (e.g. `CS-302`), credits, and numbers.
* **Reciprocal Rank Fusion (RRF)**: Applies the state-of-the-art re-ranking algorithm used by enterprise search engines to blend sparse and dense matches cleanly.
* **Query Expansion**: Utilizes the LLM to rewrite a user query into three search perspectives, capturing acronyms and synonyms to ensure zero missed context.
* **Resilient Multi-LLM Selection**: Proactively selects the available LLM client from your `.env`:
  * **Groq Cloud** (`llama-3.1-8b-instant`) — highly recommended, lightning fast, free tier.
  * **Google Gemini** (`gemini-2.5-flash`) — standard premium free tier.
  * **OpenAI** (`gpt-4o-mini`) — paid fallback.
  * **Ollama** (`llama3` local) — 100% offline fallback.
* **Strict Grounding & Citation Tracing**: Enforces inline page references (e.g., `[Page 12]`) and blocks the LLM from hallucinating when answers aren't in the B.Tech Course-book.

---

## 📁 Repository Structure

```text
├── chroma_db/             # Persistent local vector database (Git-ignored)
├── parent_store.json      # Structured parent document mapping (Git-ignored)
├── advanced_rag.py        # Standalone CLI / interactive Python application
├── advanced_rag.ipynb     # Interactive playground Jupyter Notebook
├── requirement.txt        # Full package dependencies list
├── .gitignore             # Standard git rules (excludes keys and heavy databases)
├── .env.example           # Template for environment configurations
└── README.md              # Beautiful project documentation
```

---

## 🛠️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/Vivek12042/RAGpipeline.git
cd RAGpipeline
```

### 2. Configure the Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirement.txt
```

### 3. Setup your Environment Variables
Copy the `.env.example` template into a new `.env` file and insert your API key:
```bash
cp .env.example .env
```
Open `.env` and fill in at least one key:
```text
GEMINI_API_KEY=your_active_gemini_key
# OR
GROQ_API_KEY=your_active_groq_key
```

---

## 💬 Usage Guide

### A. Standalone Python CLI
The application is pre-packaged with a command-line interface.

1. **Ask a Single Question**:
   ```bash
   python advanced_rag.py --query "What is the credit system for B.Tech?"
   ```

2. **Run Interactive Chat Mode**:
   ```bash
   python advanced_rag.py --interactive
   ```
   *(Type `exit` or `quit` to leave the chat loop)*

### B. Jupyter Notebook
Launch the `advanced_rag.ipynb` notebook inside your editor or Jupyter interface to step through individual blocks, visualize the Reciprocal Rank Fusion ranks, test different queries, and run the notebook chat interface.

---

## 🤝 Verification & Grounding Example
When asked a query that isn't present in the B.Tech Course-book, the system restricts the LLM from making up fake details:

```text
Student: What is the structure of the Summer Semester?

🔍 Processing Question: 'What is the structure of the Summer Semester?'
🔄 Expanding search query...
👉 Generated search variations: ['B.Tech Summer Semester course structure', 'Summer Semester credits course codes B.Tech', 'B.Tech Summer Semester curriculum breakdown']
📖 Performing hybrid search (Chroma + BM25) and RRF fusion...
📚 Retrieved 4 highly relevant parent chunks.
🤖 Synthesizing answer using GPT...

================================================================================
✨ ADVANCED RAG RESPONSE:
================================================================================
I cannot find the answer to the structure of the Summer Semester in the provided B.Tech Course-book. The context chunks detail the curriculum for the First Year (Semester I & II) for different sections [Page 8, Page 9] and explain the credit system [Page 4], but do not mention a Summer Semester.
================================================================================

📖 REFERENCED PAGES:
Page 4, Page 7, Page 8, Page 9
```

---

## 📜 License
This project is open-source and available under the MIT License.
