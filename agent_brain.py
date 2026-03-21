from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

class AgentState(TypedDict):
    query: str
    category: str
    response: str
    # New fields for verification
    verification_count: int 
    is_verified: bool
    feedback: str

shared_llm = Ollama(model="llama3")
embedding_model = OllamaEmbeddings(model="nomic-embed-text")

# --- NODES ---
def router_node(state: AgentState):
    print("--- ROUTING ---")
    # We tell the LLM to be very strict with its output
    prompt = f"""Categorize the following query as 'finance' or 'general'. 
    Output ONLY the word 'finance' or 'general'. 
    Query: {state['query']}"""
    
    category = shared_llm.invoke(prompt).strip().lower()
    # We check for an exact match or a clean starting word
    return {"category": "finance" if category.startswith("finance") else "general"}

def rag_node(state: AgentState):
    print(f"--- EXECUTING RAG (Attempt {state.get('verification_count', 0) + 1}) ---")    
    # 1. Get the query and any feedback from the previous loop
    current_query = state["query"]
    feedback = state.get("feedback", "")
    
    # 2. SMART SEARCH: If we failed before, rewrite the query for the auditor
    if feedback:
        print("💡 Feedback received. Optimizing search keywords...")
        # We ask Llama 3 to turn the human question into an auditor question
        search_optimizer_prompt = f"""Rewrite this user query into 3-4 professional financial keywords 
        found in a 10-K (e.g., 'Net Income', 'Balance Sheet'). Output ONLY keywords, comma-separated.
        Query: {current_query}"""
        search_query = shared_llm.invoke(search_optimizer_prompt)
    else:
        search_query = current_query

    # 3. Perform the search with the optimized query
    vector_store = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
    context_docs = vector_store.as_retriever(search_kwargs={"k": 10}).invoke(search_query)
    context_text = "\n".join([doc.page_content for doc in context_docs])
    
    # 4. Better Prompt: Force the LLM to find the table data
    prompt = f"""You are a professional Financial Auditor.
    Context: {context_text}
    
    Question: {current_query}
    Feedback from Lead Auditor: {feedback}
    
    INSTRUCTIONS:
    1. If you find financial figures, format them into a Markdown TABLE.
    2. Compare the years (e.g., 2024 vs 2025) if available.
    3. If the data is truly not in the context, say "Data not found in current index."
    
    Answer based ONLY on context:"""
    
    response = shared_llm.invoke(prompt)
    return {"response": response}

def chat_node(state: AgentState):
    print("--- EXECUTING GENERAL CHAT ---")
    return {"response": shared_llm.invoke(state["query"])}

def verifier_node(state: AgentState):
    print("--- VERIFYING RESPONSE ---")
    
    # The 'Critic' prompt
    prompt = f"""You are a financial auditor. Verify the following response for accuracy.
    1. Does it contain specific dollar amounts?
    2. Does it cite the year correctly?
    
    Response to verify: {state['response']}
    
    If it is accurate and contains numbers, respond ONLY with 'YES'.
    If it is vague or missing numbers, respond with 'NO' followed by a short reason why.
    """
    
    verification_result = shared_llm.invoke(prompt).strip()
    
    if verification_result.upper().startswith("YES"):
        return {"is_verified": True, "verification_count": state.get("verification_count", 0) + 1}
    else:
        return {
            "is_verified": False, 
            "feedback": verification_result,
            "verification_count": state.get("verification_count", 0) + 1
        }
    
def should_continue(state: AgentState):
    # Stop looping after 2 attempts to prevent infinite cycles
    if state["is_verified"] or state.get("verification_count", 0) >= 2:
        return "end"
    else:
        return "retry"
    
# --- LOGIC ---
def decide_next_node(state: AgentState):
    return state["category"] # Simply returns 'rag' or 'chat' if names match

# --- GRAPH (Build once at the end) ---
workflow = StateGraph(AgentState)

workflow.add_node("router", router_node)
workflow.add_node("rag", rag_node)
workflow.add_node("chat", chat_node)
workflow.add_node("verifier", verifier_node)

workflow.set_entry_point("router")

workflow.add_edge("rag", "verifier")

workflow.add_conditional_edges(
    "router",
    decide_next_node,
    {"finance": "rag", "general": "chat"}
)

# New: After Verifier, check if we should end or retry
workflow.add_conditional_edges(
    "verifier",
    should_continue,
    {
        "end": END,
        "retry": "rag" # Loop back!
    }
)

workflow.add_edge("chat", END)

app = workflow.compile()

# --- TEST ---
print("\n--- Testing ---")
for output in app.stream({"query": "Did Apple make money?"}):
    print(output)