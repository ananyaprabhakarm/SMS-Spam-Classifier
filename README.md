# 📩 SMS Spam Classifier

This project is a **machine learning model** for classifying SMS messages as **spam** or **ham** (not spam). It uses **natural language processing (NLP)** techniques to preprocess the text data and train a supervised learning model for accurate classification.

---

## 🧠 Features

- Preprocesses text using NLP techniques (lowercasing, punctuation removal, stopword removal, stemming)
- Converts messages into numerical vectors using **TF-IDF** or **CountVectorizer**
- Trains a classification model (e.g., **Naive Bayes**, **Logistic Regression**)
- Predicts whether incoming messages are spam or not
- Evaluates model performance with metrics like **accuracy**, **precision**, **recall**, and **F1-score**

---

## 🛠 Tech Stack

- **Python**
- **scikit-learn**
- **Pandas**
- **NumPy**
- **NLTK**
- **Matplotlib / Seaborn** (for visualizations)

---


---



## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/sms-spam-classifier.git
cd sms-spam-classifier
```
### 2. Install dependencies
```
pip install -r requirements.txt
```
3. Run the classifier
```
python spam_classifier.py
```
### ✅ Output

    Prints the classification report and confusion matrix

    Optionally visualizes data and results

    Can save the trained model for future predictions




