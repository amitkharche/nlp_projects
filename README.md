
# 🧠 NLP Projects Repository

This repository showcases end-to-end **Natural Language Processing (NLP)** projects for real-world business applications like HR automation, resume screening, sentiment analysis, and topic modeling. Each project includes a training pipeline, a Streamlit interface, and production-ready components.

---

## 📦 Included Projects

### 🤖 LangChain HR Chatbot
An automated HR assistant powered by **LangChain** and **OpenAI** to answer company policy questions using structured FAQs.

**Use Case**:
- Employee self-service
- HR helpdesk automation

**Tech**: LangChain, OpenAI embeddings, Streamlit  
**Folder**: `langchain_hr_chatbot_project/`

---

### 🧠 Resume Screening Bot
Classifies resumes based on job relevance using TF-IDF and Logistic Regression.

**Use Case**:
- Automated resume filtering
- Faster recruiter decision-making

**Tech**: TF-IDF, Logistic Regression, Streamlit  
**Folder**: `resume_screening_bot_project/`

---

### 📝 Product Review Sentiment Analysis
Analyzes customer review sentiment using supervised learning (Logistic Regression).

**Use Case**:
- Customer feedback analysis
- Product improvement planning

**Tech**: TF-IDF, Logistic Regression, Streamlit  
**Folder**: `sentiment_analysis_project/`

---

### 📚 Topic Modeling for Research Papers
Unsupervised topic modeling on academic paper abstracts using **LDA** (Latent Dirichlet Allocation).

**Use Case**:
- Discover research trends
- Aid in funding and policy decisions

**Tech**: CountVectorizer, LDA, Streamlit  
**Folder**: `topic_modeling_research_papers/`

---

## 🚀 How to Run Any Project

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Train the Model**
```bash
python model_training.py
```

3. **Launch the App**
```bash
streamlit run app.py
```

4. **Set OpenAI Key** (for LangChain projects)
```bash
export OPENAI_API_KEY="your-key-here"
```

---

## 📁 Example Project Structure

```
nlp_project/
├── data/                  # Input text or CSV files
├── model/                 # Saved models and encoders
├── app.py                 # Streamlit app
├── model_training.py      # Training logic
├── requirements.txt       # Python dependencies
├── README.md              # Project instructions
└── .github/               # Issue templates and CI
```

---

## 🧰 Tech Stack

- Python
- scikit-learn
- LangChain + OpenAI API
- TF-IDF / CountVectorizer
- Logistic Regression / LDA
- Streamlit

---

## 📄 License

All projects in this repository are licensed under the **MIT License**.

---

## 👤 Author

**Amit Kharche**  
🔗 [LinkedIn](https://www.linkedin.com/in/amitkharche)

---

## ⭐ Contributions Welcome!

If you find these projects helpful, please ⭐ the repo.  
Fork, enhance, and open a PR!

