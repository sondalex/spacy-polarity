from typing import TypedDict, Optional


class TextBlobConfig(TypedDict):
    """
    Refer to `textblob documentation <https://textblob.readthedocs.io/en/dev/api_reference.html#textblob.blob.TextBlob>`_
    """

    tokenizer: Optional[object]
    np_extractor: Optional[object]
    pos_tagger: Optional[object]
    analyzer: Optional[object]
    parser: Optional[object]
    classifier: Optional[object]


class SpacyPolarityConfig(TypedDict):
    """Whether to calculate the polarity for each sentence"""

    sentence_polarity: bool
    """Refer to :py:class:`spacy_polarity.types.TextBlobConfig`"""
    textblob_config: TextBlobConfig
