import pytest
import spacy

import spacy_polarity  # noqa: F401
from textblob import TextBlob


@pytest.fixture()
def document() -> str:
    return (
        "The financial markets are performing well, bringing good returns to investors. "
        "The stock markets in USA grew by 5% this year."
    )


@pytest.fixture()
def nlp():
    model = spacy.load("en_core_web_sm")
    model.add_pipe("spacy_polarity")
    return model


def test_component(document, nlp):
    doc = nlp(document)
    results = [sentence._.polarity for sentence in doc.sents]
    expected = list(
        map(lambda sentence: TextBlob(sentence).polarity, document.split(". "))
    )
    assert results == expected
    doc._.polarity == TextBlob(document).polarity
