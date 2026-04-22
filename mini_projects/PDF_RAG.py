from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
import google.generativeai as genai
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaEmbeddings


# 1. Load environment
load_dotenv()
embeddings_model_name = os.getenv("EMBEDDING_MODEL_NAME")
llm_model_name = os.getenv("LLM_MODEL_NAME")

# 2. Set up the model
llm = ChatGoogleGenerativeAI(
    model=llm_model_name,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
print(os.getcwd())
# 3. Load your PDF
pdf_path = os.getenv("PATH_TO_PDF")
if not pdf_path:
    # default to project-level datasets/DLHM_Final.pdf relative to this file
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    pdf_path = os.path.join(base_dir, "datasets", "DLHM_Final.pdf")
pdf_path = os.path.abspath(pdf_path)
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF not found at {pdf_path}. Set PATH_TO_PDF in .env or place the PDF there.")
loader = PyPDFLoader(pdf_path)
pages = loader.load()

# 4. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(pages)

# 5. Create embeddings + vector store
embeddings = OllamaEmbeddings(model=embeddings_model_name)
vectorstore = FAISS.from_documents(chunks, embeddings)

# 6. Create the QA chain
qa_chain1 = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant. Use the context below to answer the question.
If the context doesn't contain the answer, use your own general knowledge and say so.

Context: {context}

Question: {question}

Answer:"""
)
qa_chain2 = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": custom_prompt}

)


# 7. Ask questions in a loop!
print("PDF loaded! Ask me anything (type 'quit' to exit)\n")
while True:
    question = input("You: ")
    if question.lower() == "quit":
        break
    answer = qa_chain2.invoke(question)
    print(f"AI: {answer['result']}\n")
