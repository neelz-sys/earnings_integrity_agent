from langchain_text_splitters import RecursiveCharacterTextSplitter
import fitz

# Use the exact name of the file you downloaded
pdf_path = "_10-K-2025-As-Filed.pdf" 

# Open the document
doc = fitz.open(pdf_path)

# Print the total number of pages
print(f"Total pages: {len(doc)}")

# Load the first page (index 0)
page = doc[0]

# Extract the text from this page
text = page.get_text()

# 1. Get all text from the whole document
full_text = ""
for page in doc:
    full_text += page.get_text()

# 2. Define how to split the text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,   # Each chunk is about 1000 characters
    chunk_overlap=100  # We keep a little bit of the previous chunk for context
)

# 3. Create the chunks
chunks = text_splitter.split_text(full_text)

print(f"Total chunks created: {len(chunks)}")
print(f"First chunk snippet: {chunks[0][:200]}")

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

# 1. Initialize the embedding model
# Use this updated embedding call
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    show_progress=True  # This adds a progress bar in your terminal!
)

# 2. Create the FAISS index from your chunks
vector_store = FAISS.from_texts(chunks, embeddings)

# 3. Save it locally (this creates a folder called 'faiss_index')
vector_store.save_local("faiss_index")

print("FAISS Index created and saved!")

# 4. Test the search
query = "What were the total assets as of September 28, 2025?"
docs = vector_store.similarity_search(query, k=2)

print("\n--- Search Results ---")
for i, doc in enumerate(docs):
    print(f"Result {i+1}:\n{doc.page_content[:300]}...\n")

from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Initialize Llama 3
llm = Ollama(model="llama3")

# 2. Create a professional Prompt Template
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 3. Build the RAG Chain using the pipe (|) operator
# This translates to: "Find context -> Add question -> Send to LLM -> Parse as string"
rag_chain = (
    {"context": vector_store.as_retriever(), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 4. Ask your question
question = "Give me a short summary of Apple's total assets and liabilities for 2024 vs 2025."
print("\n--- AGENT SUMMARY ---")
print(rag_chain.invoke(question))