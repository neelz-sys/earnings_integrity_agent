import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Setup the same embedding model (the "Key" to the database)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 2. Check if the index exists before trying to load it
folder_path = "faiss_index"

if not os.path.exists(folder_path):
    print("❌ Error: No index found! Run your extraction script first.")
    exit()

print("⚡ Loading existing knowledge base...")
vector_store = FAISS.load_local(
    folder_path, 
    embeddings, 
    allow_dangerous_deserialization=True
)

# 3. Setup the Brain (Llama 3)
llm = Ollama(model="llama3")

template = """Answer the question based ONLY on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 4. The RAG Chain (The Conveyor Belt)
rag_chain = (
    {"context": vector_store.as_retriever(), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 5. Interactive Chat Loop
print("\n--- Apple 10-K Agent Ready ---")

while True:
    user_query = input("\nAsk about Apple's 10-K (or type 'exit'): ")
    
    if user_query.lower() == 'exit':
        print("Goodbye! 👋")
        break
    
    print("\nAgent Response: ", end="", flush=True)

    # We stream the response directly from the chain
    # The 'chunk' is usually just a string or a small piece of text
    for chunk in rag_chain.stream(user_query):
        print(chunk, end="", flush=True)
    
    print("\n" + "-"*30) # A visual separator for the next question