
"""
Ingest structured HR FAQs into LangChain-compatible FAISS vector store.
"""

import os
import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = "data/hr_faq.csv"
VECTOR_DIR = "model"
OPEN_AI_KEY = "sk-API Key"
# Replace with your actual key

def load_data():
    df = pd.read_csv(DATA_PATH)
    docs = [f"""Q: {row['Question']}
A: {row['Answer']}""" for _, row in df.iterrows()]
    return docs

def main():
    os.makedirs(VECTOR_DIR, exist_ok=True)
    docs = load_data()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = splitter.create_documents(docs)

    embeddings = OpenAIEmbeddings(openai_api_key=OPEN_AI_KEY)
    vectordb = FAISS.from_documents(texts, embeddings)

    vectordb.save_local(VECTOR_DIR)
    print("✅ Vector store created and saved to 'model/'")

if __name__ == "__main__":
    main()
