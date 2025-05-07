# 🔍 Turkish Hate Speech Detection with Attention-Based Deep Learning Models

It focuses on classifying Turkish tweets into one of three categories: **None (neutral)**, **Hate (hate speech)**, and **Aggressive (aggressive speech)** using deep learning models enriched with **attention mechanisms**.

---

## 📌 Project Overview

We implemented multiple classification models using **Encoder-Decoder architectures** combined with various **attention mechanisms**:

- Additive Attention (Bahdanau)
- Luong (Dot Product) Attention
- Scaled Dot-Product Attention

Two types of embeddings were tested:
- **Pre-trained Turkish Word2Vec**
- **Free (learned from scratch) embeddings**

The dataset consists of manually labeled Turkish tweets and underwent preprocessing using:
- **Zemberek** NLP for spell correction and normalization
- Class balancing via **data resampling** and **synthetic generation** using **LLaMA 3.1 7B**

---

## 🗃️ Dataset Summary

| Class        | Original Count | After Augmentation |
|--------------|----------------|---------------------|
| None         | 7667           | 2300                |
| Hate         | 2336           | 2300                |
| Aggressive   | 166            | 2300                |

Preprocessing steps included:
- UTF-8-sig encoding to preserve Turkish characters
- Removal of missing data
- Label encoding
- Padding & tokenization
- Embedding with Word2Vec

---

## ⚙️ Model Architectures

All models follow this high-level structure:

- 🔹 **Embedding Layer** (pre-trained or free)
- 🔹 **Bi-LSTM Encoder** &  **Classification Decoder** (Seq2Seq)
- 🔹 **Attention Mechanism** 
  -   Bahdanau
  -   Luong
  -   Scaled Dot-Product


---

## 📊 Results Summary

| Attention Type        | Embedding     | Accuracy | F1 Score | Inference Time |
|----------------------|---------------|----------|----------|----------------|
| Additive              | Word2Vec      | 0.827    | 0.4849   | 5.00 ms        |
| Additive              | Free          | 0.812    | 0.5322   | -              |
| Dot Product           | Word2Vec      | 0.761    | 0.2881   | 1.97 ms        |
| Dot Product           | Free          | 0.761    | 0.2881   | 0.98 ms        |
| Scaled Dot Product    | Word2Vec      | 0.726    | 0.4955   | -              |
| Scaled Dot Product    | Free          | 0.819    | 0.4859   | -              |

---

## 💡 Key Takeaways

- Pre-trained **Word2Vec embeddings** significantly improved semantic understanding.
- **Scaled Dot Product Attention** yielded the best validation performance.
- Overfitting was a challenge, especially in models with very high training accuracy.
- The **Aggressive** class remained the hardest to classify due to low sample size and semantic overlap.
- Using **equal class weights** provided more balanced results than emphasizing minority classes.

---

## 🧪 Enhancements & Experiments

- Tried **class weighting** (`[1, 1, 1]` was more stable than `[0.04, 0.94, 0.02]`)
- Ran models with **30 epochs** and **lower learning rates**
- Generated synthetic samples with **LLaMA 3.1**
- Increased dropout and number of dense layers
- Balanced the dataset with **2300 samples per class**

---

## 🧰 Technologies Used

- **Python**
- **PyTorch**
- **Gensim** (Word2Vec)
- **Zemberek** (Turkish NLP)
- **LLM**
- **Scikit-learn**
- **Matplotlib / Seaborn**

---

## 📂 Repository Structure

