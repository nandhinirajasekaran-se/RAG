# RAG
A context-aware insurance assistant that answers natural language queries by deeply understanding insurance policy documents, structured tables, and visuals.

🔄 Pipeline Stages:

Query Understanding – detects plan types (e.g., OHIP, UHIP, Enhanced) from natural language

Webase Loader – extracts structured table data and images to enhance document comprehension

Semantic Retrieval – powered by ChromaDB + BGE embeddings

Step-Back Reasoning – identifies the core concept needed to answer the question

Answer Generation – combines metadata-filtered context with LLM-based reasoning

Confidence Evaluation – scores factual alignment

Corrective Loop – expands context + regenerates answer if confidence is low

🧪 RAG Variant Exploration:
Evaluated multiple RAG strategies like:

🔹 RankRAG

🔹 FlexRAG

🔹 ChainRAG

🔹 RQ-RAG

Using RAGAs metrics, RQ-RAG showed the most accurate and stable results for insurance FAQs.
📈 Combining step-back prompting, metadata filtering, and CorrectiveRAG-style self-reflection significantly improved reliability.

⚙️ Tech Stack:
LangGraph for orchestration

LangChain, ChatOllama (LLaMA3) for LLM

ChromaDB + HuggingFace BGE embeddings

Webase loader for structured tables + images

Evaluation with RAGAs

CLI and FastAPI integration ready

📚 Example:

“Is dental coverage included under OHIP for students?”
→ The system filters for OHIP policies, extracts relevant clauses and tables, reasons through context, scores confidence, and revises if needed.
