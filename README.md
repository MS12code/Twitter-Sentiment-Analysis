# Twitter Sentiment Analysis 🐦🔍

A machine learning project that classifies tweets as **Positive**, **Negative**, or **Neutral** using the **Sentiment140 dataset** with 1.6M+ labeled tweets. Built using Python, NLP preprocessing, TF-IDF, and logistic regression.

---

## 📌 Overview

- Preprocessed tweets (tokenization, stopword removal, stemming)
- Vectorized text using TF-IDF
- Trained a logistic regression model with Scikit-learn
- Achieved **81% training accuracy** and **77.8% test accuracy**
- Visualized sentiment distribution with Matplotlib

---

## 🛠️ Tech Stack

- **Language**: Python
- **Libraries**: pandas, numpy, scikit-learn, nltk, matplotlib, tqdm

---

## 🚀 How to Run the Project

### 1. Clone the Repository

git clone https://github.com/MS12code/Twitter-Sentiment-Analysis.git
cd Twitter-Sentiment-Analysis

### 2. Install Dependencies

pip install -r requirements.txt

### 3. Download the Dataset

Download the Sentiment140 dataset from **Kaggle** and place the CSV file in the project directory.

### 4. Run the Script

python sentiment_analysis.py

## 📊 Output Example

Tweet: "I love this new update ❤️"
Predicted Sentiment: Positive


## 📈 Results
### ✅ Training Accuracy: 81%

### ✅ Test Accuracy: 77.8%

Clean performance on multiclass sentiment classification using logistic regression.

## ✨ Future Improvements
### ntegrate real-time tweet analysis with Tweepy

### Replace TF-IDF + Logistic Regression with transformer models (e.g., BERT)

### Deploy via Streamlit or Flask for web interface

## 🤝 Contributing

Contributions are welcome!
Feel free to fork this repo, raise issues, or suggest new features via pull requests.






