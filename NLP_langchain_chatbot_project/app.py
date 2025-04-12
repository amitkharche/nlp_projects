
"""
Streamlit app for HR chatbot using LangChain and OpenAI.
"""

import os
import streamlit as st
import pathlib
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

OPENAI_API_KEY = "sk-API Key"
# Replace with your actual key

st.set_page_config(page_title="🤖 HR Chatbot", layout="wide")
st.title("🤖 Ask the HR Chatbot")

@st.cache_resource
def load_qa_chain():
    base_dir = pathlib.Path(__file__).parent.resolve()
    model_path = base_dir / "model"

    if not (model_path / "index.faiss").exists():
        st.error("🚫 Vector index not found. Please run `model_training.py` first.")
        st.stop()

    vectordb = FAISS.load_local(
        str(model_path),
        OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
        allow_dangerous_deserialization=True
    )
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3, openai_api_key=OPENAI_API_KEY)
    return RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())

qa_chain = load_qa_chain()

query = st.text_input("Ask a question about HR policies:")
if query:
    result = qa_chain.run(query)
    st.write("### 🤖 Response:")
    st.write(result)
