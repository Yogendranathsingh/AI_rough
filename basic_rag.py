
import os
from typing import List
from dotenv import load_dotenv

from langchain_community.document_loader import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstrores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI


load_dotenv()

class TraditionalRAG:
    def __init__(self,model_provider: str= "groq",embedding_model: str="sentenc-transformers/all-MiniLM-L6-v2"):
        self.embedding_model=HuggingFaceEmbeddings(model_name=embedding_model)
        self.vector_store=None

        if model_provider=="groq"
            self.llm=ChatGroq(
                temperature=0,
                model_name="mixtral-8x7b-32768",
                groq_api_key=os.getenv("GROQ_API_KEY")
            )
        else:
            self.llm=ChatOpenAI(
                temperature=0,
                model_name="gpt",
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
    
    def load_documents(self, file_paths: List[str])-> List:
        documents = []
        for file_path in file_paths:
            if file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith(".txt"):
                loader = TextLoader(file_path)
            else:
                print(f"Unsupported file type: {file_path}")
                continue
        
            documents.extend(loader.load())
        return documents

    def chunk_documents(self, documents: List[str], chunk_size: int=1000,chunk_overlap: int=200) -> List[str]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
            )
        return text_splitter.split_documents(documents)
