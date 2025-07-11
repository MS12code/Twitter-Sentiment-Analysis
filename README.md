# Twitter Sentiment Analysis 🐦🔍

A machine learning project that classifies tweets as **Positive**, **Negative**, or **Neutral** using the **Sentiment140 dataset** with 1.6M+ labeled tweets. Built using Python, NLP preprocessing, TF-IDF, and logistic regression.

---

## 📌 Overview

- Preprocessed tweets (tokenization, stopword removal, stemming)
- Vectorized text using TF-IDF
- Trained a logistic regression model with Scikit-learn
- Achieved **81% training accuracy** and **77.8% test accuracy**
- Visualized sentiment distribution with Matplotlib
  - 📊 Sentiment distribution (Negative, Neutral, Positive)
  - ☁ Word clouds for each sentiment
  - 🔷 Confusion matrix
  - 🧾 Classification report (Precision, Recall, F1-score)
  - 📈 ROC curve with AUC
  - 📉 Train vs Test accuracy bar plot

---

## 🛠️ Tech Stack

- **Language**: Python
- **Libraries**:
  - `pandas`, `numpy` – Data handling
  - `nltk` – NLP preprocessing
  - `scikit-learn` – Machine learning
  - `matplotlib`, `seaborn` – Visualizations
  - `wordcloud` – Word-based visualizations
  - `tqdm` – Progress tracking

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

### ✅ AUC (ROC Curve): ≈ 0.85+

## Also includes:

Precision, recall, and F1-score for all 3 sentiment classes

ROC Curve (One-vs-Rest) to evaluate multi-class performance

Feature importance visualization for key words

Clean performance on multiclass sentiment classification using logistic regression.

## ✨ Future Improvements

Integrate real-time tweet analysis with Tweepy

Replace TF-IDF + Logistic Regression with transformer models (e.g., BERT)

Deploy via Streamlit or Flask for web interface

## 🤝 Contributing

Contributions are welcome!
Feel free to fork this repo, raise issues, or suggest new features via pull requests.






