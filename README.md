# Artificial Intelligence 2: Tweet Classification Projects

## Overview
This repository contains three advanced projects for the course "Artificial Intelligence 2," each focusing on the classification of tweets using progressively more sophisticated machine learning and deep learning techniques. The projects demonstrate a comprehensive approach to natural language processing (NLP), feature engineering, model development, and evaluation. All code is implemented in Python, leveraging libraries such as scikit-learn, PyTorch, and NLTK.

---

## Project 1: Classical Machine Learning for Text Classification

### Objective
Classify tweets into categories using classical machine learning algorithms and feature engineering.

### Data Preprocessing
- Removal of links, non-alphanumeric characters, hashtags, mentions, and stop words
- Lowercasing and lemmatization
- Feature extraction: unique word frequency, stopword frequency, average word length, POS tag frequencies

### Feature Engineering
- TF-IDF vectorization (min_df=0.01, max_df=0.4, ngram_range=(1,5), max_features=1000)
- Additional statistical and linguistic features

### Model
- Multinomial Logistic Regression (softmax)
- Hyperparameter tuning via GridSearchCV (C, penalty, solver)

### Evaluation
- Metrics: Precision, Recall, F1-score, Accuracy (per class and overall)
- Learning curve visualization

---

## Project 2: Deep Neural Networks for Text Classification

### Objective
Enhance tweet classification performance using deep neural networks (DNNs) and pre-trained word embeddings.

### Data Preprocessing
- Similar to Project 1, with custom tokenization for special tokens
- Support for GloVe pre-trained embeddings (Twitter and standard, up to 300 dimensions)

### Models
- **Model 1:**
  - Input: TF-IDF vectors
  - Architecture: [input, 1024] → Dropout(0.25) → ELU → [1024, 512] → Dropout(0.25) → ELU → [512, 3]
  - Loss: CrossEntropyLoss (label smoothing)
  - Optimizer: SGD with momentum (0.7)
  - LR Scheduler: Exponential
  - Batch size: 32, Epochs: 12
- **Model 2:**
  - Input: Mean-pooled GloVe embeddings (300d)
  - Architecture: [input, 1024] → Dropout(0.75) → LeakyReLU → [1024, 512] → Dropout(0.75) → LeakyReLU → [512, 3]
  - Loss: CrossEntropyLoss (label smoothing)
  - Optimizer: SGD with momentum (0.7), weight_decay=1e-3
  - LR Scheduler: Exponential
  - Batch size: 32, Epochs: 14

### Experiments
- Extensive testing with various optimizers (SGD, Adam, RMSprop, Adagrad, Adamax, NAdam)
- Regularization (dropout, L2)
- Embedding size comparison (25, 50, 100, 200, 300)

### Evaluation
- Metrics: Precision, Recall, F1-score, Accuracy
- ROC curve visualization
- Comparison with multinomial logistic regression

---

## Project 3: Recurrent Neural Networks for Text Classification

### Objective
Apply advanced RNN architectures (LSTM, GRU, hybrids) to sequence-based tweet classification, addressing class imbalance and sequence modeling challenges.

### Data Preprocessing
- Multiple strategies tested, including special token replacement for GloVe Twitter embeddings
- Tokenization and sequence padding

### Model Architecture
- PyTorch-based RNN with:
  - Layer 1: Bidirectional LSTM (input: embedding size, output: 40 per direction)
  - Dropout (0.2)
  - Layer 2: LSTM (input: 80, output: 3)
  - Output: Mean over sequence dimension
  - Optional attention mechanism (not used in final model)
- Input: Sequences of GloVe 300d embeddings

### Training Strategy
- Optimizer: Adam (betas=(0.8,0.9), amsgrad=True, weight_decay=1e-3)
- Loss: CrossEntropyLoss with class weights for imbalance
- Learning rate scheduler: Exponential (gamma=0.9)
- Gradient clipping (norm=3)
- Batch size: 32, Epochs: 14
- Deterministic seed setting for reproducibility

### Experiments
- Extensive hyperparameter tuning (dropout, batch size, bidirectionality, optimizer settings, gradient clipping)
- Data sorting to minimize padding
- Comparison of different preprocessing and embedding strategies
- Class imbalance handling (class weights, data augmentation discussion)

### Evaluation
- Metrics: Per-class and overall Precision, Recall, F1-score
- Learning and ROC curve visualization
- Comparison with DNN and classical models

### Key Findings
- DNN with GloVe embeddings outperformed RNNs for this dataset
- Class imbalance remains a challenge; class weighting helps but does not fully resolve minority class performance
- Advanced RNNs did not surpass simpler models, highlighting the importance of model selection based on data characteristics

---

## Reproducibility & Environment
- All code is written in Python 3.x
- Main libraries: scikit-learn, PyTorch, NLTK, pandas, numpy, matplotlib
- Pre-trained GloVe embeddings are required for Projects 2 and 3
- Notebooks are provided for each project, with clear cell structure and comments

## Contact
For questions or collaboration, please contact:
- Konstantinos Dimitrakopoulos
- Email: [Your Email Here]

---

**This repository demonstrates advanced skills in NLP, feature engineering, deep learning, and model evaluation, and is suitable for technical review by AI/ML professionals.**
