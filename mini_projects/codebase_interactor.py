#IMPORTS
from typing import List

from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_core.embeddings import Embeddings


#Load the local environment file to accesss local variables
load_dotenv()

LLM_NAME = os.getenv("LLM_NAME")
EMBEDDING_NAME = os.getenv("EMBEDDING_NAME")

LANGUAGE_MAP = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".ts": Language.TS,
    ".java": Language.JAVA,
    ".cpp": Language.CPP,
    ".go": Language.GO,
    ".rs": Language.RUST,
}

def get_llm(model_name: str, api_key: str):
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

def get_embeddings(model_name: str):
    """
    Args: model_name: the embedding models name
    Returns: an Ollama embedding model, that is stored locally
    """
    embeddings = OllamaEmbeddings(model=model_name)
    return embeddings

def get_code_reader_loader(filepath: str, embeddings: Embeddings) -> List:
    """
    Args: filepath: the path to the code file you want to load
    Return: Loader Object that can be used to create chunks of the codebase for the RAG pipeline
    """
    all_docs = []

    for ext, language in LANGUAGE_MAP.items():
        loader = GenericLoader.from_filesystem(
            path=os.getenv("CODEBASE_PATH"),
            glob=f"**/*{ext}",
            suffixes=[ext],
            parser=LanguageParser(language=language)
        )
        docs = loader.load()

        if docs:
            all_docs.extend(docs)

    return all_docs

def get_FAISS_vectorstore(loader, embeddings: Embeddings):
    """
    Args: - loader: a document loader that loads the data and splits it into chunks
          - embeddings: the embedding model that creates vector representations of the chunks
    Returns: a FAISS vectorstore that contains the vector representations of the chunks and can be
    """
    docs = loader.load()

    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore


def main() -> None:
    llm = get_llm(model_name=)