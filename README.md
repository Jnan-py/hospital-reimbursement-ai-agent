# Hospital Reimbursement AI Agent

An intelligent assistant for hospital staff that helps extract, search, and query reimbursement-related data from multiple document formats including Excel, CSV, PDF, JSON, and TXT. Powered by **Google Gemini**, **Pinecone**, and **Streamlit**, this tool enables chat-based interaction as well as one-time Q&A over uploaded tariff or billing files.

---

## Features

- Upload documents in `.xlsx`, `.csv`, `.pdf`, `.json`, `.txt` formats
- Extracts and indexes text data into Pinecone Vector Store
- Uses Gemini Pro for generating AI responses
- Context-aware chat for reimbursement-related queries
- Quick questions generated automatically based on uploaded files
- Reset session and reindex data anytime
- Streamlit-powered responsive web UI

---

## Tech Stack

- `Streamlit` – for UI
- `Google Generative AI (Gemini)` – for LLM interactions
- `LangChain` – for vector search abstraction
- `Pinecone` – for vector storage and similarity search
- `pandas`, `json`, `PyPDF2` – for parsing uploaded files
- `dotenv` – for securely managing API keys

---

## 📦 Installation

1. **Clone the repository**

```bash
git clone https://github.com/Jnan-py/hospital-reimbursement-ai-agent.git
cd hospital-reimbursement-ai-agent
```

2. **Create `.env` file**

Create a `.env` file in the root directory and add your API keys:

```env
GOOGLE_API_KEY=your_google_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the Streamlit app**

```bash
streamlit run main.py
```

---

## 🧪 Example Use Cases

- Ask: _"What is the reimbursement rate for an MRI scan in the northeast region?"_
- Explore generated quick questions from the uploaded file.
- Reset session to upload a new document and start fresh.

---

## 📁 Supported File Types

- `.xlsx`, `.xls`
- `.csv`
- `.json`
- `.txt`
- `.pdf`
