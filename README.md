# ✨ Next Word Prediction & Text Completion

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://next-word-prediction-abhijeet-ojha.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Framework](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)

A deep learning natural language processing (NLP) application that predicts the most likely next word and generates continuous text sequences using trained **LSTM** and **Simple RNN** neural networks.

---

### 🚀 Live Demo
👉 **[Launch Streamlit Web App](https://next-word-prediction-abhijeet-ojha.streamlit.app/)**

---

## ⚡ Features

- **Dual Neural Architectures:** Toggle seamlessly between **LSTM** (handles long-range context) and **Simple RNN**.
- **Top-$K$ Word Predictions:** View confidence scores and probability distributions for the most likely next words.
- **Interactive Sentence Builder:** Click **`+ Add`** on any suggested candidate token to append it to your prompt and predict subsequent words.
- **Multi-Word Text Completion:** Generate up to 25+ consecutive words with greedy or creative temperature sampling.
- **Interactive Starters:** Quick-start buttons with classic seed phrases.
- **Optimized Caching:** Cached model weights and tokenizer for fast inference.

---

## 📁 Repository Structure

```text
├── app.py                                       # Streamlit 1-page web application
├── predictor.py                                 # Inference engine & helper functions
├── lstm_model.h5                                # Trained LSTM model weights
├── rnn_model.h5                                 # Trained Simple RNN model weights
├── tokenizer.pkl                                # Keras Tokenizer vocabulary mapping
├── max_len.pkl                                  # Maximum sequence context length
├── next word prediction using RNN and LSTM.ipynb # Training & experimentation notebook
├── requirements.txt                             # Python dependencies
└── README.md
```

---

## 🛠️ Quickstart (Run Locally)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Abhijeet-ojha/next-word-prediction.git
   cd next-word-prediction
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Streamlit app:**
   ```bash
   streamlit run app.py
   ```
   Open your browser at `http://localhost:8501`.

---

## 📊 Model & Training Details

- **Dataset:** Quotes & Sayings corpus
- **Vocabulary Size:** 8,978 unique tokens
- **Max Sequence Length:** 745
- **Architecture:** Embedding Layer (100d) $\rightarrow$ Recurrent Layer (128 units) $\rightarrow$ Dense Softmax
