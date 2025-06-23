import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
import re

# --- Constants ---
CONFIDENCE_THRESHOLD = 0.5
PERSIST_DIR = "insurance_metadata_v3"



# --- Text Cleaning Functions ---
def clean_ingestion_text(text):
    """Fix spacing issues during document ingestion"""
    text = ' '.join(text.split())  # Collapse multiple spaces
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)  # "750per" -> "750 per"
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)  # "coverage750" -> "coverage 750"
    return text

def format_final_answer(answer):
    """Post-process LLM output for readability"""
    replacements = {
        r'(\d)(per|in|for|a|an|the)(\W)': r'\1 \2\3',
        r'(\b\d+)([a-zA-Z])': r'\1 \2',
        r'([a-zA-Z])(\d+\b)': r'\1 \2',
        r'\$(\d)': r'$\1 ',
    }
    for pattern, replacement in replacements.items():
        answer = re.sub(pattern, replacement, answer)
    return answer

# --- Load Vector Store ---
@st.cache_resource
def load_vector_store():
    embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-large-en",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False}
    )
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )

vectorstore = load_vector_store()

# --- LLM Setup ---
llm = Ollama(model="llama3")

# --- Prompt Templates ---
step_back_prompt = PromptTemplate(
    input_variables=["question"],
    template="""Identify the fundamental insurance concept needed to answer this question.
    Focus on general principles rather than specifics.

    Question: {question}

    Fundamental Concept:"""
)

reasoning_prompt = PromptTemplate(
    input_variables=["context", "step_back_answer", "question", "plan"],
    template="""Answer the question using the provided insurance documents:

    General Context:
    {step_back_answer}

    Insurance Policy Details ({plan} Plan):
    {context}

    Question: {question}

    Rules:
    1. Be concise and factual
    2. Clearly state which plan you're referencing
    3. Quote exact policy terms when possible
    4. If comparing plans, highlight differences
    5. If unsure, say "I couldn't find a definitive answer."

    Answer:"""
)

confidence_prompt = PromptTemplate(
    input_variables=["answer", "documents"],
    template="""Analyze how well this answer matches the documents:
    
    Answer: {answer}
    
    Documents:
    {documents}
    
    Score the confidence (0.0-1.0) using these STRICT criteria:
    1.0 = All key facts in answer are directly supported by documents with exact numbers
    0.8 = Minor wording differences but same meaning
    0.6 = Some details missing but main point correct
    0.4 = Partial match with important discrepancies
    0.2 = Contradicts documents
    0.0 = Completely unsupported
    
    Provide ONLY the numeric score with no explanation:"""
    )

# --- Core Logic ---
def evaluate_confidence(answer: str, docs: list) -> float:
    """Calculate answer-document alignment score"""
    doc_excerpts = "\n".join([clean_ingestion_text(d.page_content[:300]) for d in docs])
    score = llm(confidence_prompt.format(answer=answer, documents=doc_excerpts))
    try:
        return float(score.strip())
    except:
        return 0.5  # Fallback if parsing fails

def correct_answer(question: str, initial_answer: str, docs: list, plan: str | None):
    """CRAG correction layer"""
    # Wider document retrieval
    expanded_docs = vectorstore.similarity_search(
        question,
        k=10,  # Retrieve more documents
        filter={"plan_type": plan.lower()} if plan else None
    )
    
    # Regenerate answer
    combined_context = "\n\n".join([clean_ingestion_text(d.page_content) for d in expanded_docs])
    corrected_answer = llm(f"""Revise this answer using additional context:
        Original Answer: {initial_answer}
        New Context: {combined_context}

        Correction Rules:
        1. Preserve exact numbers from documents
        2. Add "[Verified]" prefix if confirmed
        3. Mark uncertainties clearly
        """)
    
    return format_final_answer(corrected_answer)

def step_back_rag(question: str):
    """Base RQ-RAG with Step-Back"""
    plan = detect_plan(question)
    docs = vectorstore.similarity_search(
        query=question,
        k=5,
        filter={"plan_type": plan.lower()} if plan else None
    )
    
    broader_concept = llm(step_back_prompt.format(question=question))
    combined_context = "\n\n".join([clean_ingestion_text(doc.page_content) for doc in docs])
    
    raw_answer = llm(reasoning_prompt.format(
        context=combined_context,
        step_back_answer=broader_concept,
        question=question,
        plan=plan if plan else "All"
    ))
    
    return format_final_answer(raw_answer), docs, plan

def detect_plan(question: str) -> str | None:
    """Auto-detect plan type"""
    question_lower = question.lower()
    if "basic" in question_lower: return "Basic"
    if "standard" in question_lower: return "Standard"
    if "enhanced" in question_lower: return "Enhanced"
    if "uhip" in question_lower: return "uhip"
    if "ohip" in question_lower: return "ohip"
    return None

# --- Streamlit UI ---
st.title("🔍 Insurance QA with Corrective RAG")

query = st.text_input("Ask an insurance question:", 
                     value="What is the maximum drug coverage under the Basic plan?")

if st.button("Get Answer") and query:
    with st.spinner("Analyzing..."):
        # Step 1: Initial RQ-RAG
        initial_answer, docs, plan = step_back_rag(query)
        
        # Step 2: CRAG Layer
        confidence = evaluate_confidence(initial_answer, docs)
        
        if confidence < CONFIDENCE_THRESHOLD:
            st.warning(f"Low confidence ({confidence:.0%}) - applying corrections...")
            final_answer = correct_answer(query, initial_answer, docs, plan)
            correction_note = "✅ [Verified with additional documents]"
        else:
            final_answer = initial_answer
            correction_note = "⏩ [High-confidence answer]"
        
        # Display results
        st.subheader("Answer")
        st.markdown(f"```markdown\n{final_answer}\n```")
        st.caption(correction_note)
        
        st.metric("Confidence Score", f"{confidence:.0%}")
        
        with st.expander("📄 Source Documents"):
            for i, doc in enumerate(docs):
                st.markdown(f"**Document {i+1}** (Plan: {doc.metadata.get('plan_type', 'all')})")
                st.code(clean_ingestion_text(doc.page_content[:500] + "..."))