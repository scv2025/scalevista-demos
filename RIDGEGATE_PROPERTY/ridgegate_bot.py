# Ridgegate Bot - Bot Before You Pitch
import os
import streamlit as st
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

st.set_page_config(page_title="Ridgegate Bot", layout="wide")
st.title("🤖 Live Demo: Ridgegate Bot")

openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
query = st.text_input("Ask a question", value="What services does RidgeGate offer to property investors in San Diego?")

if st.button("Run Demo"):
    if not openai_api_key:
        st.error("Please enter your OpenAI API Key.")
    else:
        os.environ["OPENAI_API_KEY"] = openai_api_key
        loader = WebBaseLoader("https://www.ridgegatepm.com/")
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        embeddings = OpenAIEmbeddings()
        db = FAISS.from_documents(chunks, embeddings)

        retriever = db.as_retriever()
        qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(temperature=0), retriever=retriever)

        response = qa.run(query)
        st.success(response)
