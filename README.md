# Public Policy RAG Demo: Multi-Document CMS Policy Retrieval (Personal Project)

This open-source prototype demonstrates retrieval-augmented generation (RAG) for federal healthcare policy research. It merges three public CMS guidance documents into a single vector index, allowing natural-language queries that return concise, grounded answers with direct citations (source file and page numbers).

Indexed documents (publicly downloadable):
- 2025 Medicaid NCCI Policy Manual: [download](https://www.cms.gov/files/document/2025nccimedicaidpolicymanualcomplete.pdf)
- 2025-2026 Medicaid Managed Care Rate Development Guide: [download](https://www.medicaid.gov/medicaid/managed-care/downloads/2025-2026-medicaid-rate-guide-082025.pdf)
- Medicaid and CHIP Managed Care Program Integrity Toolkit: [download](https://www.cms.gov/files/document/managed-care-compliance.pdf)

No private or sensitive data is used—everything draws from official federal sources.

### Purpose and Value
Healthcare policy professionals—compliance officers, analysts, consultants, and state agency staff—often need to navigate lengthy, overlapping guidance documents to locate specific rules. This demo provides fast access to relevant excerpts and clear summaries, while ensuring responses remain strictly tied to the original text for reliability in research, review, or decision-making.

### Technical Overview
Built with:
- Streamlit (frontend)
- LangChain (orchestration)
- Azure Cognitive Search (vector store)
- Local embeddings (sentence-transformers/all-MiniLM-L6-v2) for retrieval
- Azure OpenAI (gpt-4o) for generation

The hybrid design (local embeddings + cloud LLM) delivers strong semantic retrieval on technical policy text while working within free-tier constraints, such as single-deployment limits in Azure AI Foundry starter accounts.

### Running Locally
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Set required environment variables (Azure Cognitive Search endpoint/key + Azure OpenAI proxy endpoint/key + gpt-4o deployment name).
4. Run the indexing script once to load and chunk the PDFs into the Azure Search index (uses local embeddings).
5. Launch: `streamlit run app.py`

The first run downloads the local embedding model (~100 MB); subsequent runs are faster.

### Deployment to Azure App Service
- Recommended: GitHub continuous deployment.
- Configure the same environment variables in Application Settings.
- Use a Linux plan for Python/torch compatibility.
- Cold starts may include a brief model download delay.

Live demo: https://rag-demo-app-bvgthfbga2evbbeh.centralus-01.azurewebsites.net

Feedback, issues, and contributions welcome. 

This project reflects practical experience with real-world RAG challenges and is intended as a functional learning example.