# Codebase Intelligence Assistant
An AI-powered RAG (Retrieval-Augmented Generation) tool designed to index local codebases and provide an intelligent chat interface. This tool allows developers to ask complex questions about their code, understanding its structure, dependencies, and logic by leveraging Large Language Models (LLMs) and vector embeddings.

🚀 **Features**

* Language-Aware Indexing: Uses specialized parsers to understand the syntax of various programming languages, ensuring code is split into meaningful chunks.
* Multi-Language Support: Currently supports:
   * Python (`.py`)
   * JavaScript (`.js`)
   * TypeScript (`.ts`)
   * Java (`.java`)
   * C++ (`.cpp`)
   * Go (`.go`)
   * Rust (`.rs`)
* Semantic Search: Utilizes FAISS (Facebook AI Similarity Search) for high-performance vector similarity search.
* Intelligent QA: Uses a custom-engineered prompt that instructs the LLM to act as an expert assistant, focusing on code functionality, file locations, dependencies, and call chains.
* Google Gemini Integration: Powered by `ChatGoogleGenerativeAI` for state-of-the-art reasoning capabilities.

🛠️ **Tech Stack**

* Orchestration: LangChain
* LLM: Google Generative AI (Gemini)
* Vector Database: FAISS
* Embeddings: Ollama (local)
* Environment Management: `python-dotenv`

📋 **Prerequisites**

Before running the project, ensure you have:

- Python 3.10+
- [Ollama](https://ollama.com/) running locally (for embeddings)
- A Google AI Studio API key ([get one here](https://aistudio.google.com/app/apikey))

⚙️ **Setup and Installation**

1. Clone the repository:

```bash
git clone https://github.com/gergobrezina77-tech/Langchain-Projets.git
cd Langchain-Projects
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Pull your Ollama embedding model:

```bash
ollama pull nomic-embed-text
```

4. Configure Environment Variables: Create a `.env` file in the root directory and populate it with the following variables:

```env
# The name of the Google Gemini model (e.g., gemini-1.5-flash)
LLM_NAME=gemini-pro

# The name of the Ollama embedding model
EMBEDDING_NAME=nomic-embed-text

# Your Google AI Studio API Key
GOOGLE_API_KEY=your_api_key_here

# The absolute or relative path to the codebase you want to index
PATH_TO_PROJECT_FOLDER=/path/to/your/target_codebase

# The directory where the FAISS vector store will be saved
PATH_TO_VECTOR_DATABASE_FOLDER=./vector_db
```

📖 **How It Works**

1. **Indexing Phase** — The tool traverses your target directory, filtering out unnecessary folders (like `node_modules` or `__pycache__`). It uses a `LanguageParser` to read files based on their extension and splits the code into chunks of approximately 1000 characters with a 100-character overlap to maintain context. These chunks are converted into vectors and stored in a persistent FAISS index, so re-indexing is skipped on subsequent runs as long as the codebase path hasn't changed.

2. **Query Phase** — When you ask a question, the system:
   1. Converts your question into a vector embedding.
   2. Searches the FAISS database for the most relevant code snippets.
   3. Passes those snippets along with your question to the LLM using a specialized prompt.
   4. Returns a detailed answer explaining the "what," "where," and "how" of your code.

🔍 **Example Questions**

* "How does the authentication flow work in this project?"
* "What are the main dependencies of the user module?"
* "Where is the database connection initialized?"
* "Can you trace the call chain for the `process_data` function?"

⚠️ **Important Notes**

* Ensure `PATH_TO_PROJECT_FOLDER` is an absolute path to avoid indexing errors.
* The indexing process is language-dependent; adding new languages requires updating the `LANGUAGE_MAP` in the source code.
* The FAISS index is saved to `PATH_TO_VECTOR_DATABASE_FOLDER` and reused on subsequent runs — delete this folder to force a full re-index.
* The models noted in the example `.env` above are only examples; insert the appropriate model names you wish to use.
* The prompt in `create_custom_prompt()` can be freely modified to change how the assistant responds.

---

# PDF RAG — Ask Questions About Any PDF

A simple RAG pipeline that lets you chat with a PDF document. It loads a PDF, splits it into chunks, stores them in a FAISS vector store, and answers your questions using a Google Generative AI model with a custom prompt.

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) running locally (for embeddings)
- A Google AI Studio API key ([get one here](https://aistudio.google.com/app/apikey))

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Setup

1. **Clone the repo** and navigate to the project folder.

2. **Create a `.env` file** in the project root with the following variables:

```env
GOOGLE_API_KEY=your_google_api_key_here
LLM_MODEL_NAME=gemini-1.5-flash
EMBEDDING_MODEL_NAME=nomic-embed-text
PATH_TO_PDF=/absolute/or/relative/path/to/your/document.pdf
```

| Variable | Description |
|---|---|
| `GOOGLE_API_KEY` | Your Google AI Studio API key |
| `LLM_MODEL_NAME` | Google Generative AI model name (e.g. `gemini-1.5-flash`) |
| `EMBEDDING_MODEL_NAME` | Ollama embedding model name (e.g. `nomic-embed-text`) |
| `PATH_TO_PDF` | Path to the PDF you want to query. If omitted, defaults to `datasets/DLHM_Final.pdf` |

3. **Pull the embedding model** via Ollama:

```bash
ollama pull nomic-embed-text
```

---

## Usage

```bash
python mini_projects/PDF_RAG.py
```

Once loaded, you'll get an interactive prompt:

```
PDF loaded! Ask me anything (type 'quit' to exit)

You: What is this document about?
AI: ...

You: quit
```

---

## How It Works

1. Loads the PDF using `PyPDFLoader`
2. Splits it into overlapping chunks (`chunk_size=1000`, `chunk_overlap=100`)
3. Embeds the chunks locally using Ollama
4. Stores embeddings in an in-memory FAISS vector store
5. On each question, retrieves the most relevant chunks and passes them to the LLM with a custom prompt

---

## Notes

- The vector store is **not persisted** between runs — it is rebuilt each time the script starts.
- The LLM will fall back to its general knowledge if the PDF context doesn't contain the answer, and will say so.
- For large PDFs, initial embedding can take a moment depending on your machine.
- This is an amatuer PDF reader