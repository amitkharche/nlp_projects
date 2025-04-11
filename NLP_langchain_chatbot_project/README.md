# 🤖 LangChain HR Chatbot

## 📌 Use Case
Automated HR assistant chatbot to answer company policy questions using structured FAQs, powered by LangChain and OpenAI.

## 🧠 Pipeline
- Ingest FAQs into a vector store
- Use OpenAI embeddings for semantic understanding
- Deploy with Streamlit for interactive querying

## 🚀 How to Run
```bash
pip install -r requirements.txt
python model_training.py
streamlit run app.py
```

## 🔐 OpenAI Key
Set your OpenAI API key in your environment:
```bash
export OPENAI_API_KEY="your-key-here"
```
