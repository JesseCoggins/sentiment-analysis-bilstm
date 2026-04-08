# Sentiment Analysis with BiLSTM

## Overview
This project uses a Bidirectional LSTM neural network to classify movie review sentiment from text. The workflow covers text cleaning, tokenization, sequence padding, deep learning model training, and evaluation on held-out test data.

## Dataset
- Source: IMDB labeled sentences dataset
- Total reviews used: 748
- Label distribution: 362 negative, 386 positive
- Split: 598 train, 75 validation, 75 test

## Preprocessing
- lowercased text
- removed URLs and HTML
- removed punctuation and non-alphanumeric characters
- removed English stopwords with NLTK
- tokenized text with a Keras tokenizer using an OOV token
- padded all sequences to a fixed length of 50 tokens

## Model
- Embedding layer: 100 dimensions
- Bidirectional LSTM: 128 units
- Dense hidden layer: 64 units
- Dropout: 0.4 and 0.3
- Output: sigmoid for binary classification
- Early stopping on validation loss with patience of 2

## Results
- Test loss: 0.5792
- Test accuracy: 0.7600
- Precision: 0.8846
- Recall: 0.6053
- F1-score: 0.7188

## Key Takeaways
- The model achieved strong precision, meaning positive sentiment predictions were usually correct.
- Recall was lower than precision, showing the model still missed some positive examples.
- Sequence length selection mattered: a 50-token limit preserved 99.3% of the reviews while keeping the model tractable.

## Included Files
- `notebooks/Task_2_BiLSTM.ipynb`
- `data/imdb_labelled.txt`
- `data/train_dataset.csv`
- `data/validation_dataset.csv`
- `data/test_dataset.csv`
- `requirements.txt`
