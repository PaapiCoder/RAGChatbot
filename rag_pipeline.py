from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
import os


def load_documents(folder_path : str):
    docs = []
    for filename in os.listdir(folder_path):
        path = os.path.join(folder_path, filename)
        if filename.endswith(".pdf"):
            docs += PyPDFLoader(path).load()
        elif filename.endswith(".txt"):
            docs += TextLoader(path).load()
    return docs


def create_vectorstore(docs : list):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embeddings = OllamaEmbeddings(model="llama3")
    return FAISS.from_documents(chunks, embeddings)


def create_qa_chain(vectorstore : FAISS):
    llm = Ollama(model="llama3")
    retriever = vectorstore.as_retriever()
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)


# Setup once
documents = load_documents("docs")
vectorstore = create_vectorstore(documents)
qa_chain = create_qa_chain(vectorstore)


def get_answer(query : str):
    return qa_chain.run(query)
