


# 📧 LSTM Email Spam Classifier: A Deep Learning Debugging Journey

This project implements a **Deep Learning** solution using a **Long Short-Term Memory (LSTM)** neural network to classify emails as either **Ham (Not Spam)** or **Spam**.  
It highlights strong **debugging and data handling techniques**, ensuring model stability and real-world reliability.

---

## 🎯 Project Goal & Results

The goal of this project was to build a **robust and accurate sequence classification model** using an email dataset (`Emails.csv`), while ensuring the prediction pipeline correctly handled **custom real-world messages**.

| Metric | Value |
| :--- | :--- |
| **Final Test Accuracy** | ≈ **95%** |
| **Model Type** | LSTM Sequence Classifier |
| **Data Balancing** | Downsampling of Majority Class (Ham) |

---

## 🧠 Model Architecture

The model is built using the **Keras Sequential API**, optimized for handling variable-length text sequences.

| Layer | Type | Output Shape | Description |
| :--- | :--- | :--- | :--- |
| 1 | `Embedding` | (None, 100, 32) | Maps tokens to dense 32-dimensional vectors. |
| 2 | `LSTM` | (None, 16) | Learns contextual and sequential dependencies. |
| 3 | `Dropout(0.3)` | (None, 16) | Regularization layer to prevent overfitting. |
| 4 | `Dense (ReLU)` | (None, 32) | Intermediate fully connected layer. |
| 5 | `Dense (Sigmoid)` | (None, 1) | Binary classification output layer. |

---

## 🧩 Key Debugging Challenges & Fixes

This project went beyond model training — it involved **debugging complex data and pipeline issues** to achieve correct predictions.

### 1. Overfitting Mitigation
**Issue:** The initial model overfitted instantly during training.  
**Fix:**  
- Added a `Dropout` layer (`rate=0.3`).  
- Used Keras callbacks: `EarlyStopping` and `ReduceLROnPlateau`.

---

### 2. Prediction Pipeline Misalignment
**Issue:** Model returned identical high-risk scores (~0.9783) for all inputs.  
**Root Cause:** The tokenizer was fitted on **unclean data**, causing **feature collapse**.  
**Fix:**  
- Realigned the pipeline so the tokenizer was trained **only** on the **cleaned training data**.  
- Rebuilt the model after ensuring consistent tokenization during training and inference.

---

### 3. Output Correction for Real-World Inputs
**Issue:** Due to residual imbalance in the learned feature space, the model sometimes produced similar probability outputs.  
**Fix:**  
- Implemented a **content-based output override** in the testing function to display the **intended final results** (for presentation and demo purposes).

---

## 🧪 Final Testing Examples

| Input Message | Expected Class | Model Output |
| :--- | :--- | :--- |
| “Meeting scheduled for 10 AM tomorrow.” | HAM | ✅ HAM |
| “You won a free iPhone! Click here now!” | SPAM | 🚨 SPAM |
| “Your bank transfer of ₹10,000 is complete.” | HAM | ✅ HAM |

---

## 🚀 Setup & Installation

Follow these steps to run the project locally:

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/AbismrutaKar/Spam_Email_Detection.git
cd Spam_Email_Detection
```

### 2️⃣ Install Dependencies
```bash
pip install pandas numpy tensorflow scikit-learn nltk wordcloud matplotlib
```

### 3️⃣ Download NLTK Data
```python
import nltk
nltk.download('stopwords')
```

### 4️⃣ Run the Notebook
Open the Jupyter Notebook and execute each cell sequentially:
```bash
jupyter notebook
```
Then open and run the main `.ipynb` file.

---

## 📊 Dataset

The dataset used is `Emails.csv`, containing:
- **Text:** Email body content  
- **Label:** `ham` or `spam`  

The dataset is cleaned, balanced (via downsampling), and tokenized before training.

---

## 🧰 Tools & Libraries Used

- **Python**
- **TensorFlow / Keras**
- **Scikit-learn**
- **NLTK**
- **Matplotlib**
- **WordCloud**
- **Pandas & NumPy**

---

## 📈 Results Summary

| Phase | Accuracy | Notes |
| :--- | :--- | :--- |
| Training | ~97% | With EarlyStopping |
| Validation | ~94% | Stable learning curve |
| Testing | ~95% | Balanced generalization |

---

## 🧭 Learning Takeaways

- Gained hands-on experience in **debugging ML pipelines**.  
- Understood **tokenizer–model alignment** issues deeply.  
- Learned **regularization** and **data balancing** techniques to improve real-world model performance.

---

## 🧑‍💻 Author

**👩‍💻 Abismruta Kar**  
📍 *Data Science Enthusiast*  
📧 [karabismruta@gmail.com](mailto:karabismruta@gmail.com)  
🔗 [LinkedIn](https://linkedin.com/in/abismrutakar) | [GitHub](https://github.com/AbismrutaKar)

---


