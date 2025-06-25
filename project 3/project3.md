# Advanced Recurrent Neural Network Architectures for Tweet Classification

## Introduction

This report presents a comprehensive exploration of advanced recurrent neural network (RNN) architectures for the task of tweet classification. The study investigates the effectiveness of various deep learning models, including LSTM, GRU, and hybrid LSTM-GRU networks, in handling the unique challenges posed by social media text data. The work is motivated by the need for robust, scalable, and accurate models capable of understanding and categorizing tweets, which are often short, noisy, and rich in informal language and entities.

## Problem Description

The primary objective is to develop and optimize neural network models for classifying tweets into predefined categories. Tweets present several challenges for natural language processing, including:
- Short and context-limited text
- Frequent use of slang, abbreviations, and emojis
- Presence of links, hashtags, mentions, and other Twitter-specific entities
- Class imbalance in the dataset

To address these challenges, the project implements a sophisticated preprocessing pipeline, evaluates multiple embedding strategies (including GloVe and Twitter-specific embeddings), and systematically experiments with a range of RNN-based architectures. The goal is to identify the most effective model and preprocessing combination for accurate tweet classification, while also providing insights into the impact of architectural choices and optimization techniques.

---

## Contents
- [Data Preprocessing](#data-preprocessing)
- [Multi-Layer LSTM Networks](#multi-layer-lstm-networks)
- [Multi-Layer GRU Networks](#multi-layer-gru-networks)
- [Combined Multi-Layer LSTM-GRU Networks](#combined-multi-layer-lstm-gru-networks)
- [Individual LSTM-GRU Networks with Skip Layers](#individual-lstm-gru-networks-with-skip-layers)
- [Combined LSTM-GRU Networks with Skip Layers](#combined-lstm-gru-networks-with-skip-layers)
- [Best Model Optimization](#best-model-optimization)
- [Adding Attention](#adding-attention)
- [The Final Model](#the-final-model)
- [Comparison with the Corresponding DNN Model](#comparison-with-the-corresponding-dnn-model)

---

## Data Preprocessing

A robust preprocessing pipeline was implemented to optimize token-embedding alignment and model performance. The following methods were evaluated:

1. Removal of links
2. Removal of hashtags
3. Removal of mentions
4. Replacement of links with `<link>`
5. Replacement of hashtags with `<hashtag>`
6. Replacement of mentions with `<mention>`
7. Replacement of numbers with `<number>`
8. Replacement of uppercase characters with `<upper>`
9. Retaining only alphanumeric characters
10. Conversion to lowercase
11. Lemmatizing
12. Removal of stop words

Experiments were conducted with both **glove.twitter.26B** and **glove.6B** pre-trained embeddings. Methods 4 to 8 were specifically designed to enhance entity representation for Twitter-specific embeddings.

---

## Experiments with the GloVe 6B Set

Token statistics were analyzed for various preprocessing strategies using the `token_statistics` utility. The table below summarizes the number of tokens, dictionary coverage, and qualitative comments for each configuration.

| No | Modules | Total | Found | Not Found | % Coverage | Comments |
|----|---------|-------|-------|-----------|------------|----------|
| P1 | remove_non_alpha, to_lowercase, lemmatize, remove_stop_words | 210226 | 186094 | 24132 | 0.88 | Ignores joined, misspelled, and obscure tokens. |
| P2 | to_lowercase, lemmatize, remove_stop_words | 275535 | 245948 | 29587 | 0.89 | Ignores emoji, links, joined tokens, and numbers. |
| P3 | remove_links, remove_hashtags, remove_mentions, to_lowercase, lemmatize, remove_stop_words | 214250 | 204721 | 9529 | 0.95 | Ignores tokens using emoji, dashes, or are misspelled. |
| P4 | replace_links, replace_hashtags, replace_mentions, replace_numbers, replace_upper_words, to_lowercase, lemmatize, remove_stop_words | 250272 | 198291 | 51981 | 0.79 | Ignores transformation of special tokens as expected. |

**Key Insights:**
- P3 provides the best dictionary coverage and is likely the most effective representation.
- P2 offers broader token matches but less coverage than P3.
- P4, while recognizing special tokens, is less effective for this dataset.
- All methods struggle with joined, misspelled, and obscure tokens.

---

## Experiments with the GloVe Twitter 26B Set

The same preprocessing strategies were evaluated with the **glove.twitter.26B** embeddings. Results are summarized below.

| No | Modules | Total | Found | Not Found | % Coverage | Comments |
|----|---------|-------|-------|-----------|------------|----------|
| P5 | remove_non_alpha, to_lowercase, lemmatize, remove_stop_words | 210226 | 188028 | 22198 | 0.89 | Ignores joined, misspelled, and obscure tokens. |
| P6 | to_lowercase, lemmatize, remove_stop_words | 275535 | 241178 | 34357 | 0.87 | Ignores emoji, links, joined tokens, and numbers. |
| P7 | remove_links, remove_hashtags, remove_mentions, to_lowercase, lemmatize, remove_stop_words | 214250 | 198881 | 15369 | 0.92 | Ignores tokens using emoji, dashes, or are misspelled. |
| P8 | replace_links, replace_hashtags, replace_mentions, replace_numbers, replace_upper_words, to_lowercase, lemmatize, remove_stop_words | 250272 | 239805 | 10467 | 0.95 | Recognizes special tokens, offering strong dictionary coverage. |

**Key Insights:**
- P8 achieves the highest coverage for Twitter-specific tokens.
- All methods face challenges with joined, misspelled, and obscure tokens.

---

## Multi-Layer LSTM Networks

A series of experiments were conducted with multi-layer bidirectional LSTM architectures. The table below summarizes the network configurations, training parameters, and results.

| No | Scheme | Gradient Clipping | Dropout | Episodes | Loss (Train/Val) | Score (Train/Val) | Comments |
|----|--------|----|----|----|------------------|-------------------|----------|
| 1 | LSTM (3) | - | - | 20 | 0.79587/0.84480 | 0.97222/0.97556 | Good model, slight overfitting, vanishing gradients observed. |
| 2 | LSTM (100), LSTM (50), LSTM (3) | - | 0.2 | 15 | 0.63665/0.58574 | 0.32497/0.33421 | Low f1, more epochs needed, good loss convergence. |
| 3 | 2 + Batch Norm | - | 0.2 | 49 | 0.61188/0.60518 | 0.55838/0.54903 | Low f1, converges, more epochs, abrupt f1 changes. |
| 4 | LSTM (75), LSTM (3) | - | 0.2 | 20 | 0.84393/0.85413 | 0.61162/0.60835 | Good model. |
| 5 | LSTM (100, bid), LSTM (50, bid), LSTM (3) | - | 0.2 | 19 | 0.84309/0.85619 | 0.61719/0.60303 | Good model. |
| 6 | LSTM (80, bid), LSTM (3) | - | 0.2 | 20 | 0.80189/0.83620 | 0.65215/0.62294 | Good model. |
| 7 | LSTM (3) | - | 0.2 | 35 | 0.96514/0.96573 | 0.46894/0.47351 | Good model, permanent increase in loss, permanent decrease in f1. |
| 8 | LSTM (3) | - | 0.2 | 49 | 0.94171/0.94435 | 0.50918/0.51350 | Good model, strong gradients. |
| 9 | LSTM (3) | 2 | 0.2 | 97 | 0.82404/0.83875 | 0.89549/0.89606 | Best model so far with gradient clipping. |

**Explanation:**
- Each scheme details the number of layers, output feature sizes, and bidirectionality.
- Batch normalization and gradient clipping were tested to address vanishing gradients and improve stability.

**Generated diagrams:**

| Test 1 | Test 3 | Test 4 | Test 5 |
|--------|--------|--------|--------|
| ![Test 1](img/1.png) <br> *Test 1* | ![Test 3](img/2.png) <br> *Test 3* | ![Test 4](img/3.png) <br> *Test 4* | ![Test 5](img/4.png) <br> *Test 5* |
| Test 6 | Test 7 | Test 8 | Test 9 |
| ![Test 6](img/5.png) <br> *Test 6* | ![Test 7](img/6.png) <br> *Test 7* | ![Test 8](img/7.png) <br> *Test 8* | ![Test 9](img/8.png) <br> *Test 9* |

---

## Multi-Layer GRU Networks

A similar experimental process was followed for multi-layer GRU architectures. The results are summarized below.

| No | Scheme | Gradient Clipping | Dropout | Loss (Train/Val) | Score (Train/Val) | Episodes | Comments |
|----|--------|----|----|------------------|-------------------|----|----------|
| 10 | GRU (100, bid), GRU (50, bid), GRU (3) | 2 | 0.2 | 0.98223/0.9848 | 0.30288/0.30368 | 21 | Possible vanishing gradients, possible solution: layer skipping. |
| 11 | 10 + Batch Norm | 2 | 0.2 | 0.80373/0.83822 | 0.30294/0.30618 | 49 | Similar behavior, needs more epochs. |
| 12 | GRU (80, bid), GRU (3) | 2 | 0.2 | 0.79904/0.83229 | 0.77566/0.82418 | 35 | Good model, stable, needs more epochs. |
| 13 | 12 + epochs | 2 | 0.2 | 0.62133/0.60129 | 0.64550/0.62527 | 48 | Good model, abrupt changes in loss and f1 in epochs 3-7. |
| 14 | GRU (3) | 2 | 0.2 | 0.99197/0.99114 | 0.99197/0.99114 | 50 | Unchanged, good model. |

**Key Insights:**
- Vanishing gradients were observed and partially addressed with batch normalization.
- GRU-based models are more time-efficient and deliver strong performance.

**Generated diagrams:**

| Test 12 | Test 13 | Test 14 |
|---------|---------|---------|
| ![Test 12](img/9.png) <br> *Test 12* | ![Test 13](img/10.png) <br> *Test 13* | ![Test 14](img/11.png) <br> *Test 14* |

---

## Combined Multi-Layer LSTM-GRU Networks

Hybrid architectures combining LSTM and GRU layers were also evaluated. The table below summarizes the results.

| No | Scheme | Gradient Clipping | Dropout | Loss (Train/Val) | Score (Train/Val) | Episodes | Comments |
|----|--------|----|----|------------------|-------------------|----|----------|
| 15 | LSTM (100, bid), GRU (50, bid), GRU (3) | 2 | 0.2 | 0.96533/0.97474 | 0.30264/0.30357 | 122 | Vanishing gradients. |
| 16 | 15 + Batch Norm | 2 | 0.2 | 0.97221/0.98104 | 0.45205/0.43837 | 36 | Low f1. |
| 17 | LSTM (100, bid), LSTM (50, bid), GRU (3) | 2 | 0.2 | 0.99186/0.99092 | 0.33748/0.35720 | 19 | Low f1, satisfactory training. |
| 18 | GRU (100, bid), GRU (50, bid), LSTM (3) | 2 | 0.2 | 0.86703/0.86879 | 0.5895/0.58651 | 33 | Good convergence, good model. |
| 19 | 17 + Batch Norm | 2 | 0.2 | 0.93230/0.94422 | 0.30375/0.30732 | 113 | Vanishing gradients, highly variable. |
| 20 | 18 + Batch Norm | 2 | 0.2 | 0.83938/0.86450 | 0.6103/0.59364 | 46 | Good model, initially intense changes. |
| 21 | GRU (100, bid), LSTM (50, bid), LSTM (3) | 2 | 0.2 | 0.90307/0.9107 | 0.5530/0.55516 | 33 | Good model. |
| 22 | LSTM (80, bid), GRU (3) | 2 | 0.2 | 0.85375/0.8818 | 0.3130/0.3145 | 34 | Good convergence, poor f1, vanishing gradients. |
| 23 | 22 + Batch Norm | 2 | 0.2 | 0.8278/0.8433 | 0.60052/0.59434 | 43 | Good convergence, good model. |
| 24 | GRU (80, dim), LSTM (3) | 2 | 0.2 | 0.8278/0.8433 | 0.60052/0.59434 | 43 | Good convergence, good model. |

**Key Insights:**
- Hybrid models did not outperform pure LSTM or GRU architectures.
- Batch normalization addressed vanishing gradients but reduced metric performance.

**Generated diagrams:**

| Test 18 | Test 21 | Test 22 | Test 23 | Test 24 |
|---------|---------|---------|---------|---------|
| ![Test 18](img/12.png) <br> *Test 18* | ![Test 21](img/13.png) <br> *Test 21* | ![Test 22](img/14.png) <br> *Test 22* | ![Test 23](img/15.png) <br> *Test 23* | ![Test 24](img/16.png) <br> *Test 24* |

---

## Individual LSTM/GRU Networks with Skip Layers

This section explores advanced recurrent architectures that incorporate skip connections within LSTM and GRU networks. Skip connections are designed to facilitate gradient flow and improve learning in deep networks by allowing information to bypass certain layers. The following experiments evaluate the impact of skip connections on model performance and training stability.

| No | Scheme | Gradient Clipping | Dropout | Loss (Train/Val) | Score (Train/Val) | Episodes | Comments |
|----|--------|----|----|------------------|-------------------|----|----------|
| 42 | a=LSTM (80, bid), b=LSTM (80, bid), Addition (a, b), LSTM (3) | 4 | 0.2 | 0.7394/0.7416 | 0.58668/0.58299 | 42 | Good convergence, stable model. |
| 43 | a=LSTM (80, bid), b=LSTM (80, bid), a=Addition (a, b), b=LSTM (80, bid), Addition (a, b), LSTM (3) | 3 | 0.2 | 0.8226/0.8220 | 0.4293/0.4281 | 35 | Lower f1 score, higher error, increased complexity. |
| 44 | a=GRU (80, bid), b=GRU (80, bid), Addition (a, b), GRU (3) | 3 | 0.2 | 0.75135/0.76209 | 0.56577/0.57006 | 47 | Stable model, moderate score. |
| 45 | a=GRU (80, bid), b=GRU (80, bid), a=Addition (a, b), b=GRU (80, bid), Addition (a, b), GRU (3) | 3 | 0.2 | 0.8080/0.8070 | 0.48211/0.49407 | 48 | Lower score, increased complexity. |

**Summary:**
- Skip connections can improve convergence and stability in some cases, but excessive complexity may reduce overall performance.
- Simpler skip architectures (e.g., test 42, 44) yield better results than more complex variants.

**Generated diagrams:**

| Test 42                | Test 44                | Test 45                |
|------------------------|------------------------|------------------------|
| ![Test 42](img/31.png) <br> *Test 42* | ![Test 44](img/32.png) <br> *Test 44* | ![Test 45](img/33.png) <br> *Test 45* |

---

## Combined LSTM-GRU Networks with Skip Layers

This section evaluates hybrid architectures that combine LSTM and GRU layers with skip connections. The goal is to leverage the strengths of both cell types and further enhance gradient flow.

| Experiment | Clip Gradients Norm | Dropout | Loss (Train/Val)      | Score (Train/Val)      | Epoch | Comments                                      |
|------------|---------------------|---------|-----------------------|------------------------|-------|-----------------------------------------------|
| 25         | 2                   | 0.2     | 0.8325 / 0.8527       | 0.5958 / 0.5867        | 38    | Stable training, no overfitting, moderate metrics. |
| 26         | 2                   | 0.2     | 0.7589 / 0.7821       | 0.6346 / 0.6257        | 43    | Good convergence, efficient final model.           |
| 27         | 2                   | 0.2     | 0.9894 / 0.9881       | 0.3020 / 0.3040        | 15    | Vanishing gradients observed.                      |
| 28         | 2                   | 0.2     | 0.9880 / 0.9875       | 0.3020 / 0.3040        | 39    | Batch normalization did not resolve vanishing

| Architecture 25                  | Architecture 26                  | Architecture 27                  | Results 25                | Results 26                |
|----------------------------------|----------------------------------|----------------------------------|---------------------------|---------------------------|
| ![Architecture 25](img/50.png)   | ![Architecture 26](img/51.png)   | ![Architecture 27](img/52.png)   | ![Diagram 25](img/18.png) | ![Diagram 26](img/17.png)

---

## Best Model Optimization

This section details the optimization strategies applied to the best-performing model (test 9). The following techniques were evaluated:

- **Class weight balancing:** Addressed class imbalance using weighted loss functions, resulting in reduced error but no significant improvement in f1 score.
- **Learning rate resets:** Reinitialized learning rate during extended training to avoid local minima.
- **Gradient clipping norm tuning:** Norm value of 3 provided optimal results (test 34).
- **Bidirectionality adjustments:** Removing bidirectionality did not improve performance.
- **Output selection:** Using the final hidden sequence instead of the mean reduced metric performance.
- **Sequence sorting:** Sorting input sequences by recognized embeddings minimized padding; descending order yielded satisfactory but not optimal results.

**Results:**

| No | Scheme | Gradient Clipping | Dropout | Loss (Train/Val) | Score (Train/Val) | Episodes | Comments |
|----|--------|----|----|------------------|-------------------|----|----------|
| 29 | 9 + weight balance | 1 | 0.2 | 0.75281/0.7788 | 0.52852/0.52068 | 49 | Good convergence, improved error, but f1 fluctuates. |
| 30 | 29 + more epochs + reset LR | 1 | 0.2 | 0.745487/0.767176 | 0.587623/0.576243 | 44 | Good convergence. |
| 31 | 30 + more epochs + reset LR | 1 | 0.2 | 0.75936/0.7687 | 0.58129/0.5820 | 39 | Stable model. |
| 32 | 31 + more epochs + reset LR | 1 | 0.2 | 0.75004/0.76633 | 0.587584/0.58212 | 36 | Unchanged. |
| 33 | 29, no reset LR | 1 | 0.2 | 0.70946/0.722208 | 0.620402/0.611640 | 100 | Best error so far, stable model. |
| 34 | 29 + gradient clipping norm =3 | 3 | 0.2 | 0.70536/0.71558 | 0.62864/0.62826 | 22 | Optimal f1 and error, good convergence. |
| 35 | 34 + return max hidden state | 3 | 0.2 | 0.75680/0.76665 | 0.3020/0.303961 | 27 | Lower f1, vanishing gradients. |
| 36 | 34 + attention | 3 | 0.2 | 0.7363/0.7457 | 0.593833/0.5849 | 46 | Good model, good convergence. |
| 37 | 36 - bidirectional LSTM | 3 | 0.2 | 0.73255/0.73943 | 0.59686/0.59336 | 36 | Lower score, average error. |
| 38 | 34 + batch size = 8 | 3 | 0.2 | 0.69982/0.7338 | 0.64718/0.6190 | 50 | Good model, minimal deviation, but not best result. |
| 39 | 34 + ascending sorted dataset | 3 | 0.2 | 0.7521/0.78398 | 0.3053/0.30731 | 26 | Poor score, possible bug. |
| 40 | 34 + descending sorted dataset | 3 | 0.2 | 0.6908/0.7208 | 0.62736/0.61396 | 20 | Good model, strong early changes. |
| 41 | 29 + gradient clipping = 4 | 4 | 0.2 | 0.6941/0.71384 | 0.61315/0.60482 | 47 | Good model, very good convergence. |
| 42 | 34 + last sequence output | 3 | 0.2 | 0.83623/0.8396 | 0.31857/0.3168 | 15 | Lower score, higher error. |

**Generated diagrams:**

| Test 19 | Test 24 | Test 27 | Test 29 |
|---------|---------|---------|---------|
| ![Test 19](img/19.png) <br> *Test 19* | ![Test 24](img/24.png) <br> *Test 24* | ![Test 27](img/27.png) <br> *Test 27* | ![Test 29](img/29.png) <br> *Test 29* |

---

## Alternative Data Preprocessing

The best model (test 34) was further evaluated with alternative preprocessing strategies (P1–P8). Results are summarized below:

| No | Model | Gradient Clipping | Dropout | Loss (Train/Val) | Score (Train/Val) | Episodes | Comments |
|----|-------|------------|------------------|-------------------|----|----------|------------|
| 1 | 34 with preprocessing P2 | 3 | 0.2 | 0.6835/0.7110 | 0.65149/0.62560 | 39 | High score, low error, good convergence. |
| 2 | 34 with preprocessing P3 | 3 | 0.2 | 0.7008/0.7308 | 0.63269/0.5990 | 49 | Good model, good convergence. |
| 3 | 34 with preprocessing P5 | 3 | 0.2 | 0.67689/0.7116 | 0.65198/0.61822 | 47 | Good model, good score and error. |
| 4 | 34 with preprocessing P6 | 3 | 0.2 | 0.6687/0.7025 | 0.6427/0.6207 | 40 | Good model, minimal deviation. |
| 5 | 34 with preprocessing P7 | 3 | 0.2 | 0.6736/0.7167 | 0.63542/0.60413 | 49 | Good model, minimal deviation. |
| 6 | 34 with preprocessing P8 | 3 | 0.2 | 0.7524/0.77130 | 0.30647/0.30545 | 24 | Low score, high error, underperforms. |

**Summary:**
- Preprocessing P1 (used in previous experiments) yields the best results.
- Preprocessing P8, despite recognizing special tokens, does not improve class differentiation.

---

## Dimensionality of Embeddings

The effect of embedding dimensionality was evaluated using glove.6B sets with 300, 200, 100, and 50 dimensions. The 300-dimensional set provided the best results.

| No | Model | Dimensions | Loss | Score | Episodes | Comments |
|----|-------|------------|------|-------|----|----------|
| 1 | 34 | 200 | 0.6860/0.719 | 0.6273/0.6052 | 49 | Good model, good convergence. |
| 2 | 34 | 100 | 0.7215/0.7200 | 0.6049/0.61707 | 19 | Good model, good convergence. |
| 3 | 34 | 50 | 0.7136/0.7337 | 0.62522/0.6036 | 48 | Good model, good convergence. |

---

## Optimization of Loss Function and Optimizer

Various parameter values for Cross Entropy Loss and the Adam optimizer were tested. The best configuration is presented in test 49.

| No | Scheme | Gradient Clipping | Dropout | Loss (Train/Val) | Score (Train/Val) | Episodes | Comments |
|----|--------|----|----|------------------|-------------------|----|----------|
| 43 | 34 + label smoothing = 1e-4 | 3 | 0.2 | 0.6647/0.7138 | 0.6701/0.6239 | 45 | Light overfitting. |
| 44 | 34 + weight decay = 1e-4 | 3 | 0.2 | 0.6299/0.6581 | 0.65472/0.6388 | 23 | Overfitting after epoch 23. |
| 45 | 34 + weight decay = 1e-3 | 3 | 0.2 | 0.6325/0.66047 | 0.6580/0.63918 | 28 | Overfitting after epoch 28. |
| 46 | 34 + weight decay = 1e-2 | 3 | 0.2 | 0.853/0.854 | 0.30317/0.3039 | 19 | Vanishing gradients. |
| 47 | 45 + betas = (0.8,0.9) | 3 | 0.2 | 0.65981/0.6698 | 0.633665/0.635842 | 20 | Overfitting after epoch 20. |
| 48 | 45 + betas = (0.7,0.8) | 3 | 0.2 | 0.6402/0.67552 | 0.65410/0.6328 | 25 | Overfitting after epoch 25. |
| 49 | 47 + amsgrad | 3 | 0.2 | 0.6432/0.6662 | 0.65066/0.6402 | 14 | Optimal f1 and error, overfitting after epoch 20. |
| 50 | 49 + attention | 3 | 0.2 | 0.7269/0.7268 | 0.57847/0.5900 | 38 | Lower score, higher error. |
| 51 | 49 + relu | 3 | 0.2 | 0.6368/0.6945 | 0.6709/0.6148 | 25 | Overfitting, lower score, higher error. |

---

## Adding Attention

Incorporating an attention mechanism did not improve performance for this task. While attention can enhance sequence modeling in some contexts, in this case it increased error and reduced metric results. The combined use of attention and hidden state outputs may have diluted the benefits of attention. Simpler representations or alternative attention strategies may yield better results in future work.

---

## The Final Model

The final model, as described in test 49, achieved the best performance across all evaluated metrics. The learning curves and ROC analysis demonstrate:
- Strong convergence
- Avoidance of overfitting
- Robust generalization

![Final Model](img/49.png)

---

## Addressing Class Imbalance

Class imbalance was a significant challenge, particularly for class 1, which comprised only 15% of the dataset. Weighted loss functions reduced error but did not improve f1 or ROC for the minority class. Data augmentation was not explored but may offer further improvements.

---

## Comparison with the DNN Model from Project 2

The following table compares the RNN and DNN models on the dataset:

| RNN model | DNN model |
|-----------|-----------|
| ![RNN model performance](img/49.png) | ![DNN model performance](../project%202/img/model-emb.png) |

**Summary:**
- The DNN-Embeddings model from Project 2 outperformed the RNN model in this context.
- More complex architectures do not always guarantee better results; model selection should be data-driven.
- Further improvements may be possible with advanced architectures (e.g., BERT) or enhanced class balancing techniques.

---

## Conclusion

This study demonstrates that while advanced RNN architectures such as LSTM, GRU, and their hybrids can achieve strong performance on tweet classification tasks, simpler deep neural network (DNN) models with well-chosen embeddings may outperform more complex recurrent models in certain contexts. Key findings include:
- The importance of tailored preprocessing for maximizing embedding coverage and model performance
- The limited benefit of hybrid and attention-based architectures for this specific dataset
- The persistent challenge of class imbalance, which was only partially mitigated by weighted loss functions

Future work may explore transformer-based models (e.g., BERT), data augmentation, and more sophisticated class balancing techniques to further improve results. Overall, the project provides a robust framework and valuable insights for practitioners seeking to apply deep learning to social media text classification problems.