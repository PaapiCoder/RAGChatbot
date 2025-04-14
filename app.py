import streamlit as st
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
import os
import tempfile

st.set_page_config(page_title="RAG Chatbot with Upload", layout="centered")
st.title("File Upload")

# Handle file upload
uploaded_files = st.file_uploader("Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=False)

if uploaded_files:
    docs = []
    for uploaded_file in uploaded_files:
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        if suffix == ".pdf":
            docs += PyPDFLoader(tmp_file_path).load()
        elif suffix == ".txt":
            docs += TextLoader(tmp_file_path).load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embeddings = OllamaEmbeddings(model="llama3")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()
    llm = Ollama(model="llama3")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    st.success("Documents uploaded and processed successfully!")

    query = st.text_input("Ask your question")

    if query:
        with st.spinner("Thinking..."):
            response = qa_chain.run(query)
            st.success(response)
else:
    st.info("Please upload at least one document to start chatting.")