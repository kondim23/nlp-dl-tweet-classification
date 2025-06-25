# Tweet Classification: Classical, DNN, and RNN Approaches

## Overview
This repository presents a comprehensive suite of tweet classification projects, each leveraging a distinct machine learning paradigm: classical machine learning, deep neural networks (DNN), and recurrent neural networks (RNN). The codebase demonstrates practices in NLP, feature engineering, model development, and evaluation. All code is implemented in Python, using industry-standard libraries such as scikit-learn, PyTorch, and NLTK.

---

## Repository Structure

```
nlp-dl-tweet-classification/
  classical-ml-model/
      classical-ml-model-report.md
      classical-ml-tweet-classification.ipynb
  dnn-model/
      dnn-model-report.md
      dnn-tweet-classification.ipynb
  rnn-model/
      rnn-model-report.md
      rnn-tweet-classification.ipynb
  README.md/
```

- Each approach (classical ML, DNN, RNN) is organized in its own folder, containing the main report, notebook, supporting images, and extra resources.

---

## Approaches

### 1. Classical Approach
- **Techniques:** Multinomial Logistic Regression, TF-IDF, CountVectorizer, HashingVectorizer
- **Feature Engineering:** Statistical, linguistic, and POS-based features
- **Pipeline:** Modular preprocessing, vectorization, feature engineering, model selection, and evaluation
- **Evaluation:** Precision, Recall, F1-score, Accuracy, learning curves

### 2. DNN Approach
- **Techniques:** Feedforward deep neural networks
- **Embeddings:** TF-IDF and pre-trained GloVe (mean-pooled)
- **Architecture:** Two hidden layers ([1024, 512]), dropout, ELU/LeakyReLU activations
- **Optimization:** SGD/Adam, label smoothing, L2 regularization, learning rate scheduling
- **Evaluation:** Metrics, ROC curves, comparison with classical models

### 3. RNN Approach
- **Techniques:** LSTM, GRU, hybrid and skip-layer RNNs
- **Embeddings:** Pre-trained GloVe (sequence-based)
- **Architecture:** Multi-layer, bidirectional, dropout, optional attention
- **Optimization:** Adam, class weighting, gradient clipping, learning rate scheduling
- **Evaluation:** Per-class and overall metrics, learning/ROC curves, comparison with DNN and classical models

---

## Data Preprocessing
- Removal/replacement of links, hashtags, mentions, numbers, uppercase, and non-alphanumeric characters
- Lowercasing, lemmatization, stop word removal
- Custom tokenization for Twitter-specific entities
- Sequence padding for RNNs

---

## Key Findings
- DNNs with GloVe embeddings outperformed RNNs and classical models for this dataset
- Class imbalance is a persistent challenge; class weighting helps but does not fully resolve minority class performance
- Model selection should be data-driven; advanced architectures do not always guarantee better results

---

## Reproducibility & Environment
- Python 3.x
- Libraries: scikit-learn, PyTorch, NLTK, pandas, numpy, matplotlib
- Pre-trained GloVe embeddings required for DNN and RNN approaches
- Notebooks and reports provided for each approach, with clear structure and documentation
