# spacy-polarity

![pyversion](https://img.shields.io/badge/python-3.10+-blue.svg)

**Measure sentiment polarity of sentences and documents**

## Installation

```python
pip install spacy-polarity
```

## Usage

```python
import spacy
import spacy_polarity

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("spacy_polarity")

doc = nlp("The financial markets are performing well, bringing good returns to investors. The stock markets in USA grew by 5% this year.")

# Sentence level polarity  
for sent in doc.sents:
    print(sent._.polarity)
    print(type(sent._.polarity))

# Document level polarity
print(doc._.polarity)
```

## Development

### Testing

```bash
pip install ".[test]"
```

```bash
pytest -vvv
```

