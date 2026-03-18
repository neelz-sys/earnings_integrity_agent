from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

class AgentState(TypedDict):
    query: str
    category: str
    response: str

shared_llm = Ollama(model="llama3")
embedding_model = OllamaEmbeddings(model="nomic-embed-text")

# --- NODES ---
def router_node(state: AgentState):
    print("--- ROUTING ---")
    llm = Ollama(model="llama3")
    # We tell the LLM to be very strict with its output
    prompt = f"""Categorize the following query as 'finance' or 'general'. 
    Output ONLY the word 'finance' or 'general'.
    Query: {state['query']}"""
    
    category = llm.invoke(prompt).strip().lower()
    # We check for an exact match or a clean starting word
    return {"category": "finance" if category.startswith("finance") else "general"}

def rag_node(state: AgentState):
    print("--- EXECUTING RAG ---")
    
    # 1. Load the index (this is fast because it's local)
    vector_store = FAISS.load_local(
        "faiss_index", 
        embedding_model, 
        allow_dangerous_deserialization=True
    )
    
    # 2. Build the "mini-chain" inside the node
    # Note: We use the same 'shared_llm' here!
    retriever = vector_store.as_retriever()
    context_docs = retriever.invoke(state["query"])
    context_text = "\n".join([doc.page_content for doc in context_docs])
    
    prompt = f"""Answer based ONLY on this context:
    {context_text}
    
    Question: {state['query']}"""
    
    response = shared_llm.invoke(prompt)
    return {"response": response}

def chat_node(state: AgentState):
    print("--- EXECUTING GENERAL CHAT ---")
    llm = Ollama(model="llama3")
    return {"response": llm.invoke(state["query"])}

# --- LOGIC ---
def decide_next_node(state: AgentState):
    return state["category"] # Simply returns 'rag' or 'chat' if names match

# --- GRAPH (Build once at the end) ---
workflow = StateGraph(AgentState)

workflow.add_node("router", router_node)
workflow.add_node("rag", rag_node)
workflow.add_node("chat", chat_node)

workflow.set_entry_point("router")

workflow.add_conditional_edges(
    "router",
    decide_next_node,
    {"finance": "rag", "general": "chat"}
)

workflow.add_edge("rag", END)
workflow.add_edge("chat", END)

app = workflow.compile()

# --- TEST ---
print("\n--- Testing Joke ---")
for output in app.stream({"query": "What was Apple's total net sales in 2025?"}):
    print(output)