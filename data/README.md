# Sentiment data source

The review excerpts are the IMDb portion of **Sentiment Labelled Sentences**, donated by Dimitrios Kotzias (2015), [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences), DOI [10.24432/C57604](https://doi.org/10.24432/C57604). UCI states that the dataset is available under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Related paper: Kotzias, D., Denil, M., de Freitas, N., and Smyth, P. (2015). *From Group to Individual Labels Using Deep Features*, KDD.

This project uses `imdb_labelled.txt`, the IMDb source file from the UCI dataset cited above. Reading quotation marks literally yields 1,000 labeled records, with 500 in each class. Three exact duplicate text/label pairs are excluded from modeling. Identical cleaned texts stay within one split: 799 training, 100 validation and 98 test records.

The split CSVs contain padded token sequences and labels. The tokenizer was fitted only on training reviews. `test_predictions.csv` pairs source-row indices with held-out labels, probabilities and predictions; `evaluation_metrics.json` and `training_history.csv` record the evaluation and learning curves. This is movie-review sentiment data, not customer-support feedback collected for this project. Dataset attribution and license are separate from the code's MIT license.
