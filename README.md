# sentiment-analysis-lstm-bert-comparison
Comparative Sentiment Analysis project using Custom LSTM, Pretrained AWD-LSTM (ULMFiT), and BERT on the IMDb dataset. Built with PyTorch and HuggingFace, this project evaluates recurrent and transformer-based architectures based on accuracy, convergence speed, and contextual understanding with detailed performance analysis.
# 📊 Sentiment Analysis Using LSTM, AWD-LSTM (ULMFiT) & BERT

A comprehensive NLP project comparing Custom LSTM, Pretrained AWD-LSTM (ULMFiT), and Transformer-based BERT models for sentiment classification using the IMDb Movie Reviews dataset.

---

## 🚀 Project Overview

This project implements and compares three deep learning architectures for binary sentiment classification:

- Custom LSTM (trained from scratch)
- Pretrained AWD-LSTM (ULMFiT methodology)
- Transformer-based BERT (bert-base-uncased)

The objective is to analyze how transfer learning and transformer architectures improve performance over traditional recurrent neural networks.

---

## 🎯 Problem Statement

Design and implement a sentiment analysis system using:

1. A custom LSTM model trained from scratch  
2. A pretrained AWD-LSTM model (ULMFiT)  
3. A BERT transformer model  

Compare performance in terms of:

- Accuracy
- Precision
- Recall
- F1-score
- Convergence speed
- Generalization ability

---

## 📂 Dataset

**IMDb Movie Reviews Dataset**

- Source: Stanford AI Lab / HuggingFace
- 50,000 labeled reviews
- Binary classification (Positive / Negative)
- Long-form text reviews suitable for sequence modeling

---

## 🛠 Technologies Used

- Python
- PyTorch
- HuggingFace Transformers
- TorchText / Tokenizers
- NumPy
- Matplotlib
- Deep Learning (RNN & Transformers)

---

## 🧠 Models Implemented

### 1️⃣ Custom LSTM
- Built from scratch in PyTorch
- Embedding Layer
- LSTM Layers
- Fully Connected Layer
- Trained without pretrained weights

### 2️⃣ AWD-LSTM (ULMFiT)
- Transfer learning approach
- Fine-tuned pretrained language model
- Faster convergence
- Improved accuracy on limited data

### 3️⃣ BERT (bert-base-uncased)
- Transformer-based architecture
- WordPiece tokenization
- Self-attention mechanism
- Fine-tuned classification head
- Superior contextual understanding

---

## 📊 Evaluation Metrics

All models evaluated using identical metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Training Loss
- Validation Loss
- Epochs to Convergence

---

## 📈 Comparative Results

| Model        | Accuracy | Convergence Speed | Context Handling | Computational Cost |
|-------------|----------|------------------|------------------|-------------------|
| Custom LSTM | Moderate | Slow             | Basic            | Low               |
| AWD-LSTM    | High     | Faster           | Good             | Moderate          |
| BERT        | Highest  | Fastest          | Excellent        | High              |

### Key Findings

- Transfer learning significantly improves performance.
- AWD-LSTM converges faster than custom LSTM.
- BERT achieves the highest accuracy.
- Transformers handle long-range dependencies better than RNNs.
- BERT requires higher computational resources.

---

## 📁 Project Structure


├── Sentiment_Analysis_Final_Project.ipynb # Custom LSTM + AWD-LSTM
├── Sentiment_BERT.ipynb # BERT Implementation
├── models/ # Saved model weights
├── results/ # Performance plots
├── README.md


---

## ⚙️ How to Run

### Install Dependencies

```bash
pip install torch torchvision transformers datasets numpy matplotlib

Run LSTM & AWD-LSTM Notebook
jupyter notebook Sentiment_Analysis_Final_Project.ipynb

Run BERT Notebook
jupyter notebook Sentiment_BERT.ipynb

👨‍💻 Author

Mukul
Deep Learning & NLP Enthusiast
