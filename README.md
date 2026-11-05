# Movie-review sentiment and feedback triage

I built a bidirectional LSTM to classify short IMDb reviews as positive or negative. A possible use is to send likely complaints to a person for follow-up. On **98 test reviews**, the classifier flagged **47** as negative. That queue contained **40 of the 54 negative reviews** and **7 positive reviews**. A support team can use those counts to judge the reading workload and the complaints it would miss.

![Held-out sentiment confusion matrix: 40 negative and 37 positive reviews classified correctly](assets/sentiment-test-confusion.png)

*Confusion matrix for 98 held-out reviews; rows show actual sentiment and columns show predictions.*

[Open the evaluation notebook](notebooks/sentiment_bilstm.ipynb) · [Inspect the trained model and tokenizer](models/README.md) · [View the source reviews](data/imdb_labelled.txt)

## Results

| Actual sentiment | Predicted negative | Predicted positive |
|---|---:|---:|
| Negative | 40 | 14 |
| Positive | 7 | 37 |

The model correctly classified **77 of 98 reviews (78.57%)**. With *positive* as the target class, precision was **72.55%**, recall **84.09%** and F1 **0.7789**, at a 0.5 threshold. Always predicting the more common test class would score 55.10%. The test set is small, so these results need confirmation on a larger independent set. The negative-review queue caught 40 of 54 negative reviews and missed 14. [Paired predictions](data/test_predictions.csv) · [Evaluation metrics](data/evaluation_metrics.json)

## Text processing and model

The source contains **1,000 labeled review excerpts**, read as tab-separated records with literal quotation marks. I removed three exact duplicate text/label pairs, leaving **997 records**. Stratified group folds keep identical cleaned texts in the same partition: **799 training, 100 validation and 98 test reviews**. Grouping makes these approximately 80/10/10 splits.

I lowercased text, removed URLs, HTML and punctuation, and removed English stopwords. The tokenizer was fitted on training text only; unseen words map to an out-of-vocabulary token. The 50-token input limit covered all cleaned training reviews. The model used a 100-dimensional embedding, a 128-unit bidirectional LSTM, a 64-unit dense layer, dropout and early stopping. Training completed five epochs and restored epoch 3, which had the lowest validation loss. [Review lengths](assets/review-lengths.png) · [Training curves](assets/training-history.png)

The IMDb evaluation describes movie-review sentiment. A customer-support pilot would need evaluation on feedback from its intended channel. This study did not measure response-time savings or customer outcomes.

## Run it yourself

Install the [Python requirements](requirements.txt) in a Jupyter environment and run the notebook with `notebooks/` as its working directory. The first run requests the NLTK stopwords resource if it is unavailable. A fixed seed and deterministic TensorFlow operations support rerunning the workflow; numerical environments can still affect fitted weights.

To use the saved model without retraining, follow the [inference example](models/README.md). The tokenizer and preprocessing metadata are included. The notebook reloads the model and tokenizer and verifies that they reproduce all held-out predictions. [Training split](data/train_dataset.csv) · [Validation split](data/validation_dataset.csv) · [Test split](data/test_dataset.csv)

## Data and environment

Data: IMDb excerpts from UCI Sentiment Labelled Sentences; [attribution](data/README.md).  
Install: `python -m pip install -r requirements.txt`; [environment details](requirements-verified.txt).

*WGU M.S. in Data Analytics coursework, D604 Advanced Analytics.*
