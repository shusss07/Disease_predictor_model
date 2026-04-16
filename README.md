# Disease Predictor from Symptoms

A machine learning model that predicts diseases based on natural language symptom descriptions. Built with TensorFlow/Keras and scikit-learn, trained on 1,200 symptom-disease pairs across 24 disease categories.

## Results

| Metric | Score |
|---|---|
| Test Accuracy | **97%** |
| Macro Precision | 0.97 |
| Macro Recall | 0.97 |
| Macro F1-Score | 0.97 |
| Test Samples | 240 |


## Overview

This project takes a plain-English description of symptoms (e.g. *"I have a skin rash and joint pain"*) and predicts the most likely disease. It uses TF-IDF vectorization to convert text into features, and a neural network classifier to make predictions.

---

## Dataset

- **File:** `Symptom2Disease.csv` from kaggle https://www.kaggle.com/datasets/niyarrbarman/symptom2disease
- **Size:** 1,200 rows × 2 columns (`label`, `text`)
- **Classes:** 24 diseases
- **Split:** 80% train / 20% test
- **Format:** Each row is a free-text symptom description paired with a disease label

---

## Model Architecture

```
Preprocessing(punctuation_removal,stopwords_removal,stemming)
        ↓
Input (TF-IDF features)
        ↓
Dense(128, activation='relu')
        ↓
Dropout(0.5)
        ↓
Dense(64, activation='relu')
        ↓
Dense(24, activation='softmax')
```

- **Optimizer:** Adam
- **Loss:** Sparse Categorical Crossentropy
- **Epochs:** 20

---

## Pipeline

1. **Preprocessing** — Lowercasing, stemming (NLTK), stopword removal
2. **Vectorization** — TF-IDF (`TfidfVectorizer` from scikit-learn)
3. **Label Encoding** — `LabelEncoder` converts disease names to integers
4. **Train/Test Split** — 80/20 split
5. **Training** — Keras Sequential neural network
6. **Evaluation** — Accuracy, classification report

---

## Getting Started

### Prerequisites

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib nltk
```

### Clone the repo

```bash
git clone https://github.com/shusss07/Disease_predictor_model
cd Disease_predictor_model
```

### Run the notebook

Open `Model.ipynb` in Jupyter and run all cells. Make sure `Symptom2Disease.csv` is in the same directory.

---

## Making a Prediction

```python
def predict_disease(symptom_text):
    vec = vectorizer.transform([symptom_text]).toarray()
    pred = model.predict(vec)
    disease = encoder.inverse_transform([pred.argmax()])
    return disease[0]

predict_disease("I have a skin rash and joint pain")
# Output: 'psoriasis'
```

---

## Project Structure

```
disease-predictor/
│
├── Model.ipynb           # Main notebook
├── Symptom2Disease.csv   # Dataset
└── README.md
```

---

## Disclaimer

This tool is for **educational purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical concerns.

---

## Built With

- [TensorFlow / Keras](https://www.tensorflow.org/)
- [scikit-learn](https://scikit-learn.org/)
- [pandas](https://pandas.pydata.org/)
- [NLTK](https://www.nltk.org/)
- [Matplotlib](https://matplotlib.org/)