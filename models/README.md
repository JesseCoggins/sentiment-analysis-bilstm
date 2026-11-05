# Trained sentiment classifier

The Keras model, training-fitted tokenizer and preprocessing metadata form one inference package. The model accepts 50 padded token IDs and returns the probability of positive sentiment. The metadata includes the exact English stopword list used in training. The notebook verifies that reloading these artifacts reproduces its held-out predictions.

Run this from the repository root after installing the requirements:

```python
import json
import re
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

directory = Path("models")
metadata = json.loads((directory / "preprocessing.json").read_text(encoding="utf-8"))
stop_words = set(metadata["english_stopwords"])
tokenizer = tokenizer_from_json((directory / "tokenizer.json").read_text(encoding="utf-8"))
model = load_model(directory / "imdb_sentiment_bilstm_model.keras", compile=False)

def clean_text(text):
    text = re.sub(r"http\S+|www\S+", " ", str(text).lower())
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(word for word in text.split() if word not in stop_words)

reviews = ["I enjoyed the performances and the story."]
inputs = pad_sequences(
    tokenizer.texts_to_sequences([clean_text(text) for text in reviews]),
    maxlen=metadata["max_sequence_length"],
    padding=metadata["padding"], truncating=metadata["truncating"],
)
probabilities = model(inputs, training=False).numpy().reshape(-1)
for probability in probabilities:
    label = "Positive" if probability >= metadata["threshold"] else "Negative"
    print(label, float(probability))
```

The probability is the model's score, not a calibrated guarantee. See the [evaluation](../README.md#results) and [source notes](../data/README.md).
