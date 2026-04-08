Started This porject : 18 March 7:17PM for techfest showcase

# 🧠 MindScan — Mental Health Detection from Text

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-Llama_3.1-412991?style=flat-square)


**MindScan detects mental health distress signals in written text using NLP and connects users to Mello 🫧 — an AI support chatbot that already knows what you shared.**

🔗 **Live Demo:** [mindscanwithmello.streamlit.app](https://mindscanwithmello.streamlit.app)

---

## What it does

You type something — a journal entry, how you're feeling, anything — and MindScan tells you what mental health signals it detects. If you want to talk about it, Mello opens and already has context from what you wrote.

**7 categories detected:** Normal · Depression · Anxiety · Stress · Suicidal · Bipolar · Personality Disorder

If suicidal signals are detected, Mello opens automatically with Indian crisis helpline numbers.

---

---

## Tech Stack

| What | Tool |
|---|---|
| ML model | scikit-learn |
| Web app | Streamlit |
| AI chatbot | Groq (Llama 3.1) |
| Charts | Plotly |
| Data | pandas |
---


## How it works

```
Your text → TF-IDF converts it to numbers → Logistic Regression predicts category
                                                        ↓
                                           Confidence score + bar chart
                                                        ↓
                                           Mello reads your text + mood
                                                        ↓
                                           Context-aware chat begins
```

**Model:** TF-IDF (15k features) + Logistic Regression · **Accuracy:** ~78% · **Dataset:** 53,000+ Reddit posts

## Run locally

```bash
# Clone and setup
git clone https://github.com/YOUR_USERNAME/mental-health-detector.git
cd mental-health-detector
python -m venv venv
.\venv\Scripts\Activate.ps1      # Windows
pip install -r requirements.txt

# Add your Groq API key (get free key at console.groq.com)
echo GROQ_API_KEY=your_key_here > .env

# Train the model then run
python train.py
streamlit run streamlit_app.py


```

---

## Project files

```
├── train.py           ← trains and saves the ML model
├── streamlit_app.py   ← the entire web app
├── app.py             ← FastAPI backend (optional)
├── data/              ← dataset CSV (not in Git)
└── models/            ← saved model.pkl
```

---





