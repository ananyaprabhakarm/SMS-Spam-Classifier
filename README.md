# 📩 SMS Spam Classifier

This project is a **machine learning model** for classifying SMS messages as **spam** or **ham** (not spam). It uses **natural language processing (NLP)** techniques to preprocess the text data and train a supervised learning model, served through a **Streamlit** web app.

---

## 🧠 Features

- Preprocesses text using NLP techniques (lowercasing, tokenization, stopword removal, stemming)
- Converts messages into numerical vectors using **TF-IDF**
- Classifies messages with a trained **Multinomial Naive Bayes** model
- Simple Streamlit UI for entering a message and getting an instant prediction

---

## 🛠 Tech Stack

- **Python**
- **Streamlit**
- **scikit-learn**
- **NLTK**
- **Pandas / NumPy / Matplotlib / Seaborn** (for the exploratory data analysis and model training in `main.ipynb`)

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/ananyaprabhakarm/SMS-Spam-Classifier.git
cd SMS-Spam-Classifier
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run main.py
```

This opens the app in your browser, where you can paste a message and click **Predict** to see whether it's classified as **Spam** or **Not Spam**.

---

## 📁 Project Structure

- `main.ipynb` — data cleaning, EDA, and model training/selection
- `main.py` — Streamlit app that loads the trained model and serves predictions
- `spam.csv` — the SMS spam dataset used for training
- `vectorizer.pkl` / `model.pkl` — the fitted TF-IDF vectorizer and classifier used by the app
