import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_community.embeddings import HuggingFaceEmbeddings  # Local

# Env for Search only (no OpenAI needed for embeddings)
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")

pdf_urls = [
    "https://www.cms.gov/files/document/2025nccimedicaidpolicymanualcomplete.pdf",
    "https://www.medicaid.gov/medicaid/managed-care/downloads/2025-2026-medicaid-rate-guide-082025.pdf",
    "https://www.cms.gov/files/document/managed-care-compliance.pdf"
]

docs = []
for url in pdf_urls:
    loader = PyPDFLoader(url)
    docs.extend(loader.load())

print(f"Loaded {len(docs)} pages.")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)
print(f"Split into {len(chunks)} chunks.")

# Local embeddings
local_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Vector store (uses local for embedding_function)
vector_store = AzureSearch(
    azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
    azure_search_key=AZURE_SEARCH_KEY,
    index_name="rag-index",
    embedding_function=local_embeddings.embed_query  # Local
)

vector_store.add_documents(chunks)
print("Indexing complete with local embeddings.")