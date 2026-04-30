# 📰 Truth Lens — AI Fake News Detection System

## 📌 Overview
Truth Lens is a machine learning–based web application that detects whether a news article is **REAL or FAKE** using Natural Language Processing (NLP) techniques.

The system analyzes text patterns, word frequency, and statistical features to make predictions using multiple ML models.

---

## 🚀 Features
- 🔍 Detects Fake vs Real news
- 🤖 Uses multiple ML models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
- 📊 Ensemble-based final prediction (improved accuracy)
- ⚡ Real-time prediction using Streamlit UI
- 🎯 Confidence scores for each model

---

## 🧠 Tech Stack
- Python
- Streamlit
- Scikit-learn
- Pandas
- NLP (TF-IDF Vectorization)

---

## 📂 Project Structure

app.py # Main Streamlit application
Fake.csv # Fake news dataset
True.csv # Real news dataset
README.md # Project documentation


---

## ▶️ How to Run the Project

### Step 1: Install dependencies

pip install -r requirements.txt


### Step 2: Run the application

streamlit run app.py


### Step 3: Open in browser

http://localhost:8501


---

## 🧪 How to Use
1. Paste any news article into the input box  
2. Click **DETECT →**  
3. View predictions from multiple models  
4. Check final verdict (REAL / FAKE)

---

## ⚠️ Limitations
- Model depends on training data patterns
- May misclassify unseen or complex news articles
- Not a replacement for professional fact-checking

---

## 🔮 Future Improvements
- Integration with live news APIs
- Explainable AI (highlight influential words)
- Deep learning models (BERT)
- Real-time fact-checking sources

---

## 📌 Final Step (Important for Reviewers)

To fully experience the project:

👉 Run the application locally or access the deployed version (if provided)  
👉 Use sample or real-world news articles to test predictions  
👉 Observe model confidence and final ensemble decision  

---

## 👨‍💻 Author
**Macwin Roy**

---

## 📎 Note
This project is developed for academic and learning purposes to demonstrate the use of machine learning in fake news 