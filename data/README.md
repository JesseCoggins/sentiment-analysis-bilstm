# Sentiment data source

The review excerpts are the IMDb portion of **Sentiment Labelled Sentences**, donated by Dimitrios Kotzias (2015), [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences), DOI [10.24432/C57604](https://doi.org/10.24432/C57604). UCI states that the dataset is available under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Related paper: Kotzias, D., Denil, M., de Freitas, N., and Smyth, P. (2015). *From Group to Individual Labels Using Deep Features*, KDD.

`imdb_labelled.txt` is the source file used in WGU D604. The notebook’s tab-separated parser yields 748 review records. The three CSVs contain derived padded token sequences and labels (598 training, 75 validation, 75 test); the notebook documents cleaning and tokenization. This is movie-review sentiment data, not customer-support feedback collected for this project. Dataset attribution and license are separate from the code’s MIT license.
