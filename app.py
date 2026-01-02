import os
import streamlit as st
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

st.set_page_config(page_title="Public Policy RAG Demo: Azure Multi-Doc CMS Retrieval", layout="centered")

st.markdown("""
<style>
    .block-container {max-width: 900px; padding-top: 1rem; padding-bottom: 3rem;}
    .main {background-color: #ffffff;}
    body {color: #1e3a8a; font-family: 'Arial', sans-serif; font-size: 1.2rem; line-height: 1.8;}
    h1, h2, h3 {color: #1e3a8a;}
    .stTextInput > div > div > input {font-size: 1.2rem;}
    .stChatMessage {font-size: 1.2rem;}
</style>
""", unsafe_allow_html=True)

st.title("Public Policy RAG Demo: Multi-Document CMS Policy Retrieval (Personal Project)")

st.markdown("""
### About This Demo
Personal open-source project exploring retrieval-augmented generation (RAG) for policy and compliance research.  
Answers grounded exclusively in three public CMS documents (freely downloadable):
- 2025 Medicaid NCCI Policy Manual (coding edits/guidelines): [CMS link](https://www.cms.gov/files/document/2025nccimedicaidpolicymanualcomplete.pdf)
- 2025-2026 Medicaid Managed Care Rate Development Guide (rate setting): [CMS link](https://www.medicaid.gov/medicaid/managed-care/downloads/2025-2026-medicaid-rate-guide-082025.pdf)
- Medicaid and CHIP Managed Care Program Integrity Toolkit (compliance tools): [CMS link](https://www.cms.gov/files/document/managed-care-compliance.pdf)

No private, confidential, or personal data used—pure public federal guidance available to anyone. Not affiliated with or endorsed by any government agency.

#### Why Multi-Document RAG
Professionals working with federal healthcare policy (e.g., state compliance officers, analysts, or consultants) often need to quickly locate and understand specific rules in lengthy official documents—like coding edits, guidelines, or implementation details—to support accurate decision-making or rate adjustments.  
This prototype merges the three sources into one searchable index, retrieving relevant, cited excerpts across documents for broader insights.

**Intended Value**: Faster access to precise information from official sources, aiding general research, policy review, or professional workflows—while ensuring responses stay strictly tied to the original text for reliability.

#### How to Use 🔍
Ask natural-language questions. Responses provide factual summaries + expandable source excerpts with file/page.

**Try these examples** (copy-paste):
- What is the National Correct Coding Initiative? (NCCI manual)
- How are capitation rates developed in managed care? (Rate Guide)
- Compare NCCI PTP edits and program integrity requirements. (cross NCCI + Integrity Toolkit)
- Summarize rate adjustments and medically unlikely edits (MUEs). (cross Rate Guide + NCCI)
- Explain modifiers in NCCI and their relation to compliance tools. (cross all three)

Built with LangChain + Azure AI Search + Azure OpenAI + App Service.
""")

# Azure config from env vars
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt4o")

required_vars = [
    AZURE_SEARCH_ENDPOINT,
    AZURE_SEARCH_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_KEY
]

if not all(required_vars):
    missing = [name for name, val in {
        "AZURE_SEARCH_ENDPOINT": AZURE_SEARCH_ENDPOINT,
        "AZURE_SEARCH_KEY": AZURE_SEARCH_KEY,
        "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT,
        "AZURE_OPENAI_KEY": AZURE_OPENAI_KEY
    }.items() if not val]
    st.error(f"Missing required Azure credentials: {', '.join(missing)}. Check your environment variables.")
    st.stop()

# Local embeddings for retrieval
local_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Cloud LLM
llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT,
    temperature=0.05,
    api_version="2025-01-01-preview"  # Matches your successful test
)

vector_store = AzureSearch(
    azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
    azure_search_key=AZURE_SEARCH_KEY,
    index_name="rag-index",
    embedding_function=local_embeddings.embed_query
)

retriever = RunnableLambda(lambda query: vector_store.hybrid_search(query, k=12))

prompt = PromptTemplate.from_template("""
You are an expert on CMS Medicaid policy documents. Answer using only the provided context excerpts from the manuals. 
If not covered, say "Not covered in the provided excerpts."
Cite file names and page numbers.

Context:
{context}

Question: {question}

Answer:
""")

def format_docs(docs):
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = os.path.basename(doc.metadata.get('source', 'unknown'))
        page = doc.metadata.get('page', 'unknown') + 1
        formatted.append(f"Excerpt {i} ({source}, Page {page}):\n{doc.page_content}\n")
    return "\n".join(formatted)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("Retrieved excerpts"):
                st.markdown(message["sources"])

if prompt := st.chat_input("Ask about policy rules across CMS documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Retrieving & generating..."):
            response = chain.invoke(prompt)
            sources = format_docs(retriever.invoke(prompt))
            st.markdown(response)
            if sources:
                with st.expander("Retrieved excerpts"):
                    st.markdown(sources)
                st.session_state.messages.append({"role": "assistant", "content": response, "sources": sources})
            else:
                st.session_state.messages.append({"role": "assistant", "content": response})

st.caption("Personal open-source project—feedback welcome! Public documents only.")