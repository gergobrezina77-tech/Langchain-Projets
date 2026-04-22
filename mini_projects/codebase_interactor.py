#IMPORTS
import json
from typing import List
from langchain_core.vectorstores import VectorStore        # parent of FAISS
from langchain_core.language_models import BaseLanguageModel  # parent of ChatGoogleGenerativeAI
from langchain_classic.chains.retrieval_qa.base import Chain                    # parent of RetrievalQA

from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
import sys

sys.stdout.reconfigure(encoding='utf-8')

#Load the local environment file to accesss local variables
load_dotenv()

LLM_NAME = os.getenv("LLM_NAME")
EMBEDDING_NAME = os.getenv("EMBEDDING_NAME")
API_KEY=os.getenv("GOOGLE_API_KEY")
CODEBASE_PATH = os.getenv("PATH_TO_PROJECT_FOLDER")
VECTOR_DB = os.getenv("PATH_TO_VECTOR_DATABASE_FOLDER")
LANGUAGE_MAP = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".ts": Language.TS,
    ".java": Language.JAVA,
    ".cpp": Language.CPP,
    ".go": Language.GO,
    ".rs": Language.RUST,
}



def get_llm(model_name: str, api_key: str) -> BaseLanguageModel:
    """
    Args: model_name: the name of the LLM model you want to use
            api_key: the API key to access the model, if needed

    Note: This can be changed to any other LLM, here a Google Generative model is used
    from Google AI studio, for changing, import the right class and change the code accordingly
    """
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key
    )
    return llm


def get_embeddings(model_name: str) -> Embeddings:
    """
    Args: model_name: the embedding models name
    Returns: an Ollama embedding model, that is stored locally
    """
    embeddings = OllamaEmbeddings(model=model_name)
    return embeddings


def get_code_reader_loader(filepath: str) -> List:
    """
    Args: filepath: the path to the code file you want to load
    Return: docs object that that is a list containing the chunks of the code files
    """
    all_docs = []

    for ext, language in LANGUAGE_MAP.items():
        for root, dirs, files in os.walk(filepath):
            # skip hidden and environment folders
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
            
            # check if this folder has any files with this extension
            matching = [f for f in files if f.endswith(ext)]
            if not matching:
                continue
                
            try:
                loader = GenericLoader.from_filesystem(
                    path=root,
                    glob=f"*{ext}",
                    suffixes=[ext],
                    parser=LanguageParser(language=language, parser_threshold= 0)
                )
                loaded = loader.load()
                all_docs.extend(loaded)
            except Exception as e:
                print(f"Skipping {root} ({ext}): {e}")

    return all_docs


def save_vectorstore_meta(codebase_path: str, vectorstore_path: str) -> None:
    meta = {"codebase_path": codebase_path}
    os.makedirs(vectorstore_path, exist_ok=True)
    meta_file = os.path.join(vectorstore_path, "meta.json")
    with open(meta_file, "w") as f:
        json.dump(meta, f)


def is_vectorstore_valid(codebase_path: str, vectorstore_path: str, meta_path: str) -> bool:

    # vectorstore_path must be an existing directory containing FAISS index
    if not os.path.isdir(vectorstore_path):
        return False
    # FAISS expects an index file inside the folder (commonly 'index.faiss')
    index_file = os.path.join(vectorstore_path, "index.faiss")
    if not os.path.exists(index_file):
        print(f"FAISS index missing: {index_file}")
        return False
    if not os.path.exists(meta_path):
        return False
    
    with open(meta_path, "r") as f:
        meta = json.load(f)
    
    return meta.get("codebase_path") == codebase_path

def split_documents(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)
    return chunks

def get_FAISS_vectorstore(docs: List, embeddings: Embeddings, path: str) -> VectorStore:
    """
    Args: - docs: a list that contains the chunks that should be stored in the vector database
          - embeddings: the embedding model that creates vector representations of the chunks
    Returns: a FAISS vectorstore that contains the vector representations of the chunks and can be
    """
    if os.path.exists(path):
        vectorstore = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents(docs, embeddings)
        os.makedirs(path, exist_ok=True)
        vectorstore.save_local(path)

    return vectorstore


def create_custom_prompt() -> PromptTemplate:
    """
    Returns: a custom prompt template that can be used in the QA chain
    Note: You can change to prompt to reflect your wishes and to make to model behave differentaly
    """
    custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""This is a codebase, with possibly multiple programming languages. You are an assistant, and you are asked questions about this codebase, stored in the context. 
    When answering pay attention to:
    - What the code does
    - Where it lives in the codebase
    - What dependencies it has
    - How it interacts with other parts of the codebase
    - What call chain could be behind it

    In your answer assume a person who is intermediate in coding. 
    You may also use your external knowledge to answer theoretical programming or library related questions.

    Context: {context}

    Question: {question}

    Answer:"""
    )
    return custom_prompt


def create_qa_chain(model: BaseLanguageModel, vectorstore: VectorStore, prompt: PromptTemplate) -> Chain:
    qa_chain = RetrievalQA.from_chain_type(
        llm=model,
        retriever = vectorstore.as_retriever(search_type="mmr", 
                                             search_kwargs={"k": 8, "fetch_k": 20}),
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain


def get_vectorstore_path(codebase_path: str) -> str:
    base_folder = VECTOR_DB
    # Normalize codebase_path and derive a safe project name
    project_name = os.path.basename(os.path.normpath(codebase_path))
    if not project_name or project_name in (".", os.sep):
        # fallback: use a sanitized absolute path as a name
        abs_path = os.path.abspath(codebase_path)
        project_name = abs_path.lstrip(os.sep).replace(os.sep, "_")
        if not project_name:
            project_name = "project"

    return os.path.join(base_folder, project_name)

def main() -> None:
    if not CODEBASE_PATH:
        raise RuntimeError(
            "Environment variable PATH_TO_PROJECT_FOLDER is empty. "
            "Set PATH_TO_PROJECT_FOLDER in your .env to the project you want to index (absolute or relative path)."
        )
    if not os.path.exists(CODEBASE_PATH):
        raise RuntimeError(f"PATH_TO_PROJECT_FOLDER does not exist: {CODEBASE_PATH}")

    vectorstore_path  = get_vectorstore_path(CODEBASE_PATH)
    meta_path  = os.path.join(vectorstore_path, "meta.json")

    llm = get_llm(model_name=LLM_NAME, api_key=API_KEY)

    embeddings = get_embeddings(model_name=EMBEDDING_NAME)

    if is_vectorstore_valid(CODEBASE_PATH, vectorstore_path, meta_path):
            vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
    else:
        docs = get_code_reader_loader(filepath=CODEBASE_PATH)
        chunks = split_documents(docs)
        vectorstore = get_FAISS_vectorstore(chunks, embeddings, vectorstore_path)
        os.makedirs(vectorstore_path, exist_ok=True)
        vectorstore.save_local(vectorstore_path)
        save_vectorstore_meta(CODEBASE_PATH, vectorstore_path)


    prompt = create_custom_prompt()

    qa_chain = create_qa_chain(model=llm, vectorstore=vectorstore, prompt=prompt)

    while True:
        question = input("You: ")
        if question.lower() == "quit":
            break
        answer = qa_chain.invoke(question)
        print(f"AI: {answer['result']}\n")


if __name__ == "__main__":
    main()