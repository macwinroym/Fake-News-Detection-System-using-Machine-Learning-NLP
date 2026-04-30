import streamlit as st
import pandas as pd
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Truth Lens",
    page_icon="📰",
    layout="wide"
)

# --------------------------------------------------
# CSS
# --------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800&display=swap');

body {
    background-color: #0b0f17;
    color: white;
}

.brand {
    font-family: 'Playfair Display', serif;
    font-size: 64px;
    font-weight: 800;
    color: #e60000;
}

.big-title {
    text-align: center;
    font-size: 40px;
    font-weight: 700;
}

.red {
    color: #e60000;
}

.sub-text {
    text-align: center;
    color: #aaa;
    margin-bottom: 30px;
}

textarea {
    background-color: #1c2230 !important;
    color: white !important;
}

.stButton>button {
    background-color: #e60000;
    color: white;
    font-weight: bold;
}

.result-box {
    background-color: white;
    color: black;
    border-radius: 10px;
    padding: 25px;
    text-align: center;
}

.fake {
    color: red;
    font-weight: bold;
}

.real {
    color: green;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown('<div class="brand">Truth Lens</div>', unsafe_allow_html=True)
st.markdown('<div class="big-title">Expose the Narrative.</div>', unsafe_allow_html=True)
st.markdown('<div class="big-title red">Uncover the Truth.</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">AI Fake News Detection Systems</div>', unsafe_allow_html=True)

# --------------------------------------------------
# LOAD DATA (BALANCED FIX)
# --------------------------------------------------
@st.cache_data
def load_data():
    fake = pd.read_csv("Fake.csv")
    true = pd.read_csv("True.csv")

    fake["label"] = 0
    true["label"] = 1

    # Balance dataset
    min_len = min(len(fake), len(true))
    fake = fake.sample(min_len, random_state=42)
    true = true.sample(min_len, random_state=42)

    data = pd.concat([fake, true], ignore_index=True)
    data = data.sample(frac=1, random_state=42)

    return data[["text", "label"]]

data = load_data()

# --------------------------------------------------
# CLEAN TEXT
# --------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r'\n', '', text)
    return text

data["text"] = data["text"].apply(clean_text)

# --------------------------------------------------
# SPLIT
# --------------------------------------------------
x = data["text"]
y = data["label"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# --------------------------------------------------
# TF-IDF
# --------------------------------------------------
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7, max_features=5000)
xv_train = vectorizer.fit_transform(x_train)

# --------------------------------------------------
# MODELS (FIXED LR)
# --------------------------------------------------
@st.cache_resource
def train_models():
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    dt = DecisionTreeClassifier()
    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    lr.fit(xv_train, y_train)
    dt.fit(xv_train, y_train)
    rf.fit(xv_train, y_train)

    return lr, dt, rf

LR, DT, RF = train_models()

# --------------------------------------------------
# INPUT
# --------------------------------------------------
news = st.text_area("", placeholder="Paste your news article...", height=200)

# --------------------------------------------------
# DETECTION
# --------------------------------------------------
if st.button("DETECT →"):

    if news.strip() == "":
        st.warning("Please enter some news text.")
    else:
        cleaned_news = clean_text(news)
        news_vec = vectorizer.transform([cleaned_news])

        pred_lr = LR.predict(news_vec)[0]
        pred_dt = DT.predict(news_vec)[0]
        pred_rf = RF.predict(news_vec)[0]

        prob_lr = LR.predict_proba(news_vec)[0][1]
        prob_dt = DT.predict_proba(news_vec)[0][1]
        prob_rf = RF.predict_proba(news_vec)[0][1]

        # Weighted decision (FIX)
        final_score = (prob_lr * 0.5) + (prob_rf * 0.3) + (prob_dt * 0.2)
        final_pred = 1 if final_score > 0.55 else 0

        col1, col2, col3 = st.columns(3)

        def show_result(col, title, prob, pred):
            with col:
                st.markdown(f"""
                <div class='result-box'>
                    <h4>{title}</h4>
                    <h2>{round(prob*100,2)}%</h2>
                    <p class='{'real' if pred==1 else 'fake'}'>
                        {'REAL' if pred==1 else 'FAKE'}
                    </p>
                </div>
                """, unsafe_allow_html=True)

        show_result(col1, "Logistic Regression", prob_lr, pred_lr)
        show_result(col2, "Decision Tree", prob_dt, pred_dt)
        show_result(col3, "Random Forest", prob_rf, pred_rf)

        st.markdown("###")

        st.markdown(f"""
        <div class='result-box'>
            <h2>Final Verdict</h2>
            <h1 class='{'real' if final_pred==1 else 'fake'}'>
                {'REAL NEWS' if final_pred==1 else 'FAKE NEWS'}
            </h1>
        </div>
        """, unsafe_allow_html=True)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.caption("Truth Lens • AI Fake News Detection Systems")