from abc import ABCMeta, abstractmethod
from typing import Optional
from spacy.language import Language
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span
from textblob import TextBlob
from spacy_polarity.types import TextBlobConfig, SpacyPolarityConfig
from typing import Iterable

from spacy_polarity._version import __version__


class PolarityABC(metaclass=ABCMeta):
    def __init__(
        self,
        has_sentencizer: bool,
        tokenizer: Optional[object] = None,
        np_extractor: Optional[object] = None,
        pos_tagger: Optional[object] = None,
        analyzer: Optional[object] = None,
        parser: Optional[object] = None,
        classifier: Optional[object] = None,
    ):
        self.tokenizer = tokenizer
        self.np_extractor = np_extractor
        self.pos_tagger = pos_tagger
        self.analyzer = analyzer
        self.parser = parser
        self.classifier = classifier
        self.__post__init__()

    @abstractmethod
    def __call__(self, doc: Doc) -> Doc: ...

    @abstractmethod
    def __post__init__(self): ...


class HierarchicalPolarity(PolarityABC):
    def __call__(self, doc: Doc) -> Doc:
        sentences: Iterable[Span] = doc.sents
        for sentence in sentences:
            textblob = TextBlob(
                sentence.text,
                tokenizer=self.tokenizer,
                np_extractor=self.np_extractor,
                pos_tagger=self.pos_tagger,
                analyzer=self.analyzer,
                parser=self.parser,
                classifier=self.classifier,
            )

            sentence._.polarity = textblob.polarity
        doc._.polarity = TextBlob(doc.text).polarity
        return doc

    def __post__init__(self):
        if not Doc.has_extension("polarity"):
            Doc.set_extension("polarity", default=None)
        if not Span.has_extension("polarity"):
            Span.set_extension("polarity", default=None)


def hierarchical_polarity(
    nlp: Language,
    name: str,
    sentence_polarity: bool,
    textblob_config: TextBlobConfig,
    tokenizer=None,
    np_extractor=None,
    pos_tagger=None,
    analyzer=None,
    parser=None,
    classifier=None,
):
    has_sentencizer = "senter" in nlp.component_names and sentence_polarity
    return HierarchicalPolarity(has_sentencizer, **textblob_config)


DEFAULT_CONFIG: SpacyPolarityConfig = {
    # Whether to apply sentence polarity
    "sentence_polarity": True,
    "textblob_config": {
        "tokenizer": None,
        "np_extractor": None,
        "pos_tagger": None,
        "analyzer": None,
        "parser": None,
        "classifier": None,
    },
}


Language.factory(
    "spacy_polarity",
    default_config=DEFAULT_CONFIG,
    func=hierarchical_polarity,
)


__all__ = ["__version__", "DEFAULT_CONFIG", "HierarchicalPolarity"]
