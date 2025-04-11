"""
Streamlit app for HR chatbot using LangChain and OpenAI.
"""

import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

st.set_page_config(page_title="🤖 HR Chatbot", layout="wide")
st.title("🤖 Ask the HR Chatbot")

@st.cache_resource
def load_qa_chain():
    vectordb = FAISS.load_local("model", OpenAIEmbeddings())
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
    return RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())

qa_chain = load_qa_chain()

query = st.text_input("Ask a question about HR policies:")
if query:
    result = qa_chain.run(query)
    st.write("### 🤖 Response:")
    st.write(result)
