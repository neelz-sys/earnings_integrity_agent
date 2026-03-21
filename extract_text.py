import fitz
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

# --- CONFIGURATION ---
PDF_PATH = "_10-K-2025-As-Filed.pdf"
INDEX_NAME = "faiss_index"

# 1. Page-by-Page Extraction
print(f"📂 Extracting documents from: {PDF_PATH}")
doc = fitz.open(PDF_PATH)
documents = []

for i, page in enumerate(doc):
    # We create a 'Document' for every page to keep metadata intact
    page_doc = Document(
        page_content=page.get_text(),
        metadata={"page": i + 1} # Adding 1 to make it human-readable
    )
    documents.append(page_doc)

# 2. Document-Aware Chunking
# Instead of split_text, we use split_documents to preserve that metadata
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)
print(f"📦 Created {len(chunks)} chunks with page metadata.")

# --- 3. EMBEDDING & SAVING (Memory Optimized) ---
print("\n🧬 Initializing Embedding Model...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# We define a batch size (e.g., 50 chunks at a time)
batch_size = 50
vector_store = None

print(f"🏗️ Building FAISS Index in batches of {batch_size}...")

for i in range(0, len(chunks), batch_size):
    batch = chunks[i : i + batch_size]
    
    if vector_store is None:
        # Create the initial index
        vector_store = FAISS.from_documents(batch, embeddings)
    else:
        # Add to the existing index
        vector_store.add_documents(batch)
    
    print(f"✅ Indexed chunks {i} to {min(i + batch_size, len(chunks))}...")

print(f"💾 Saving index to: '{INDEX_NAME}'")
vector_store.save_local(INDEX_NAME)