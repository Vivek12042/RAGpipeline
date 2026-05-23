import os
import sys
import json
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Configuration constants
PDF_PATH = "/Users/vivekrajpurohit/practicemodel/ragmodel /Course-book_B.Tech_._summer2025-revised.pdf"
PERSIST_DIR = "/Users/vivekrajpurohit/practicemodel/ragmodel /chroma_db"
PARENT_STORE_PATH = "/Users/vivekrajpurohit/practicemodel/ragmodel /parent_store.json"

def get_llm():
    """
    Dynamically initializes the LLM based on available API keys in the environment.
    Supports OpenAI, Groq, Gemini, and local Ollama.
    """
    # 1. Groq (Recommended Free Tier Cloud provider)
    if os.environ.get("GROQ_API_KEY"):
        try:
            from langchain_groq import ChatGroq
            print("🚀 Using Groq Cloud LLM (Llama 3)...")
            return ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)
        except ImportError:
            print("📦 Installing langchain-groq library...")
            import subprocess
            subprocess.run([sys.executable, "-m", "pip", "install", "langchain-groq"], check=True)
            from langchain_groq import ChatGroq
            print("🚀 Using Groq Cloud LLM (Llama 3)...")
            return ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)

    # 2. Google Gemini
    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        if os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            print("🚀 Using Google Gemini LLM (gemini-2.5-flash)...")
            return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
        except ImportError:
            print("📦 Installing langchain-google-genai library...")
            import subprocess
            subprocess.run([sys.executable, "-m", "pip", "install", "langchain-google-genai"], check=True)
            from langchain_google_genai import ChatGoogleGenerativeAI
            print("🚀 Using Google Gemini LLM (gemini-2.5-flash)...")
            return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)

    # 3. OpenAI (Standard fallback, but prone to quota limits)
    if os.environ.get("OPENAI_API_KEY") and not os.environ.get("OPENAI_API_KEY").startswith("sk-proj-expired"):
        try:
            from langchain_openai import ChatOpenAI
            print("🚀 Using OpenAI LLM (gpt-4o-mini)...")
            return ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        except ImportError:
            print("📦 Installing langchain-openai library...")
            import subprocess
            subprocess.run([sys.executable, "-m", "pip", "install", "langchain-openai"], check=True)
            from langchain_openai import ChatOpenAI
            print("🚀 Using OpenAI LLM (gpt-4o-mini)...")
            return ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

    # 4. Fallback to local Ollama (Llama 3)
    print("🤖 No cloud API keys found. Falling back to local Ollama (llama3)...")
    try:
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(model="llama3", temperature=0.0)
    except ImportError:
        try:
            from langchain_ollama import ChatOllama
            return ChatOllama(model="llama3", temperature=0.0)
        except ImportError:
            print("📦 Installing langchain-community for Ollama support...")
            import subprocess
            subprocess.run([sys.executable, "-m", "pip", "install", "langchain-community"], check=True)
            from langchain_community.chat_models import ChatOllama
            return ChatOllama(model="llama3", temperature=0.0)

def load_and_index_pdf():
    """
    Checks if index exists on disk. If so, loads it.
    Otherwise, parses PDF, performs Parent-Child splitting, and indexes chunks.
    """
    # Load free local embeddings (runs completely offline on your Mac CPU)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Check if files already exist
    if os.path.exists(PERSIST_DIR) and os.path.exists(PARENT_STORE_PATH):
        print("⚡ Loading existing persistent indices from disk...")
        vectorstore = Chroma(
            collection_name="btech_child_chunks",
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings
        )
        with open(PARENT_STORE_PATH, "r") as f:
            parent_store = json.load(f)
        
        # Reconstruct parent Document objects for BM25
        parent_docs = []
        for pid, data in parent_store.items():
            parent_docs.append(Document(
                page_content=data["content"],
                metadata=data["metadata"]
            ))
        
        print(f"✅ Loaded {len(parent_store)} parent chunks and Vector database successfully!")
        return vectorstore, parent_store, parent_docs

    print("🚀 Ingestion Pipeline Started: Parsing B.Tech Course-book PDF...")
    if not os.path.exists(PDF_PATH):
        print(f"❌ ERROR: PDF file not found at {PDF_PATH}")
        sys.exit(1)
        
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()
    print(f"📄 Loaded {len(pages)} pages from the PDF.")

    # 1. Define splitters
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

    parent_store = {}
    parent_docs = []
    child_docs = []
    
    parent_idx = 0
    print("✂️ Splitting pages into Parent and Child chunks...")
    
    for page in pages:
        page_num = page.metadata.get("page", 0) + 1  # 1-indexed page number
        
        # Split page into parent chunks
        parents = parent_splitter.split_documents([page])
        for p in parents:
            parent_id = f"parent_{parent_idx}"
            parent_data = {
                "id": parent_id,
                "content": p.page_content,
                "metadata": {
                    "parent_id": parent_id,
                    "page": page_num,
                    "source": "Course-book_B.Tech"
                }
            }
            parent_store[parent_id] = parent_data
            
            p_doc = Document(page_content=p.page_content, metadata=parent_data["metadata"])
            parent_docs.append(p_doc)
            
            # Split parent chunk into child chunks
            children = child_splitter.split_text(p.page_content)
            for c_text in children:
                child_docs.append(Document(
                    page_content=c_text,
                    metadata={
                        "parent_id": parent_id,
                        "page": page_num
                    }
                ))
            parent_idx += 1

    print(f"📦 Created {len(parent_docs)} parent chunks and {len(child_docs)} child chunks.")
    print("🧠 Generating embeddings and writing to Chroma DB (this may take a moment)...")
    
    # Create and persist Chroma Vector DB
    vectorstore = Chroma.from_documents(
        documents=child_docs,
        embedding=embeddings,
        collection_name="btech_child_chunks",
        persist_directory=PERSIST_DIR
    )
    
    # Save parent store to disk
    with open(PARENT_STORE_PATH, "w") as f:
        json.dump(parent_store, f, indent=4)
        
    print("🎉 Ingestion complete! Databases written to disk.")
    return vectorstore, parent_store, parent_docs

def expand_query(query, llm):
    """
    Expands the user query into 3 alternative formulations to increase retrieval recall.
    """
    prompt = f"""You are an advanced search system assistant. 
Your task is to generate 3 alternative search queries based on the user's input query.
These queries should be optimized for a search engine retrieving academic syllabi, course codes, course structures, credits, and semesters from a B.Tech course book.
Do not write any introductory or concluding text, just list the 3 queries, one per line.

User Query: {query}

Alternative Queries:"""
    
    try:
        response = llm.invoke(prompt)
        queries = [q.strip().strip("-* ").strip() for q in response.content.split("\n") if q.strip()]
        # Add original query to the search list
        queries.insert(0, query)
        # Filter empty strings and deduplicate
        unique_queries = list(dict.fromkeys([q for q in queries if q]))
        return unique_queries[:4]  # Return original + 3 alternatives
    except Exception as e:
        print(f"⚠️ Query expansion failed, using original query. Error: {e}")
        return [query]

def reciprocal_rank_fusion(dense_results, sparse_results, k=60):
    """
    Combines dense and sparse search results using Reciprocal Rank Fusion (RRF).
    """
    rrf_scores = {}
    
    # Process dense results
    for rank, doc in enumerate(dense_results):
        parent_id = doc.metadata["parent_id"]
        if parent_id not in rrf_scores:
            rrf_scores[parent_id] = {"doc": doc, "score": 0.0}
        rrf_scores[parent_id]["score"] += 1.0 / (k + (rank + 1))
        
    # Process sparse results
    for rank, doc in enumerate(sparse_results):
        parent_id = doc.metadata["parent_id"]
        if parent_id not in rrf_scores:
            rrf_scores[parent_id] = {"doc": doc, "score": 0.0}
        rrf_scores[parent_id]["score"] += 1.0 / (k + (rank + 1))
        
    # Sort documents descending based on their RRF score
    sorted_items = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in sorted_items]

def perform_hybrid_search(expanded_queries, vectorstore, parent_store, bm25, top_n=5):
    """
    Runs both Dense (Chroma child matching -> parent retrieval) and Sparse (BM25 on parent chunks)
    across all expanded queries, merging matches using Reciprocal Rank Fusion (RRF).
    """
    dense_parents_retrieved = []
    sparse_parents_retrieved = []
    
    for q in expanded_queries:
        # 1. Dense Search: retrieve child chunks, map to parent documents
        child_matches = vectorstore.similarity_search(q, k=6)
        for child in child_matches:
            pid = child.metadata["parent_id"]
            parent_data = parent_store[pid]
            dense_parents_retrieved.append(Document(
                page_content=parent_data["content"],
                metadata=parent_data["metadata"]
            ))
            
        # 2. Sparse Search: retrieve parent documents directly
        sparse_matches = bm25.invoke(q)[:4]
        sparse_parents_retrieved.extend(sparse_matches)
        
    # Apply RRF to fuse all results
    fused_docs = reciprocal_rank_fusion(dense_parents_retrieved, sparse_parents_retrieved)
    
    # Return top N unique parent chunks
    return fused_docs[:top_n]

def generate_answer(query, context_docs, llm):
    """
    Compiles context docs and sends query to LLM with a strict grounded system prompt.
    """
    context_text = ""
    for idx, doc in enumerate(context_docs):
        page = doc.metadata.get("page", "Unknown")
        context_text += f"--- [Document Chunk {idx+1} | Source Page: {page}] ---\n"
        context_text += f"{doc.page_content}\n\n"
        
    system_prompt = """You are an expert academic counselor and guide for the B.Tech program.
Your task is to answer the student's question based strictly and only on the provided course book context chunks below.

Guidelines for answering:
1. Ground every statement strictly in the provided context. If the context does not contain the answer, say clearly: "I cannot find the answer in the provided B.Tech Course-book." Do not make up facts.
2. For EVERY claim or piece of information you write, you MUST include a clean inline citation citing the exact page number of the source chunk in the format '[Page X]'. 
   Example: 'Students must complete 160 credits to graduate [Page 12].'
3. Maintain a highly professional, academic, and structured tone. Use markdown bullet points and tables where appropriate to present credit maps, syllabi, or rules clearly.
4. When listing courses, always include course codes (e.g., CS-302) if they are present in the text.
"""

    prompt = f"""{system_prompt}

Retrieved Context Chunks:
=======================================
{context_text}
=======================================

Student Query: {query}

Synthesized Answer (remember to cite exact Page numbers inline):"""

    response = llm.invoke(prompt)
    return response.content

def run_rag(query, vectorstore, parent_store, bm25, llm):
    """
    Helper to run full Advanced RAG pipeline and print results.
    """
    print(f"\n🔍 Processing Question: '{query}'")
    
    # 1. Query Expansion
    print("🔄 Expanding search query...")
    expanded = expand_query(query, llm)
    print(f"👉 Generated search variations: {expanded[1:]}")
    
    # 2. Hybrid Retrieval with RRF
    print("📖 Performing hybrid search (Chroma + BM25) and RRF fusion...")
    retrieved = perform_hybrid_search(expanded, vectorstore, parent_store, bm25, top_n=4)
    print(f"📚 Retrieved {len(retrieved)} highly relevant parent chunks.")
    
    # 3. Grounded Answer Synthesis
    print("🤖 Synthesizing answer using GPT...")
    answer = generate_answer(query, retrieved, llm)
    
    # Print results beautifully
    print("\n" + "="*80)
    print("✨ ADVANCED RAG RESPONSE:")
    print("="*80)
    print(answer)
    print("="*80)
    
    print("\n📖 REFERENCED PAGES:")
    pages_referenced = sorted(list(dict.fromkeys([doc.metadata["page"] for doc in retrieved])))
    print(", ".join([f"Page {p}" for p in pages_referenced]))
    print("="*80 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Advanced RAG for B.Tech Course Book")
    parser.add_argument("--query", type=str, help="Single question to query RAG pipeline")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive command-line mode")
    args = parser.parse_args()

    # Load/Build indices
    vectorstore, parent_store, parent_docs = load_and_index_pdf()
    
    # Initialize BM25 Sparse Retriever
    bm25 = BM25Retriever.from_documents(parent_docs)
    
    # Initialize LLM (Auto-Provider Select)
    llm = get_llm()
    
    if args.query:
        run_rag(args.query, vectorstore, parent_store, bm25, llm)
    elif args.interactive:
        print("\n🎓 B.Tech Course-book RAG Assistant is Ready! (Type 'quit' or 'exit' to exit)")
        print("-" * 65)
        while True:
            try:
                user_q = input("💬 Ask a question about the course book: ")
                if user_q.lower() in ["quit", "exit"]:
                    break
                if not user_q.strip():
                    continue
                run_rag(user_q, vectorstore, parent_store, bm25, llm)
            except KeyboardInterrupt:
                break
    else:
        # Default dry run query
        default_query = "What is the structure of the Summer Semester?"
        run_rag(default_query, vectorstore, parent_store, bm25, llm)

if __name__ == "__main__":
    main()
